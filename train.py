#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from cityscapes import CityScapes
from gta5 import GTA5
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
import argparse

logger = logging.getLogger()


def val(args, model, dataloader):
    print("start val!")
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print("precision per pixel for test: %.3f" % precision)
        print("mIoU for validation: %.3f" % miou)
        print(f"mIoU per class: {miou_list}")

        return precision, miou


def train(args, model, optimizer, dataloader_train, dataloader_val):
    print("start train")
    # for SummaryWriter read (https://pytorch.org/docs/stable/tensorboard.html)
    writer = SummaryWriter(
        log_dir="/content/Drive/MyDrive/AML project/logs",
        comment="".format(args.optimizer),
    )  # log is in run/ folder

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0

    for epoch in range(args.epoch_start_i + 1, args.num_epochs + 1):
        lr = poly_lr_scheduler(
            optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs
        )
        model.train()  # Sets module in training mode
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description("epoch %d, lr %f" % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()  # Zero-ing the gradients

            with amp.autocast():
                output, out16, out32 = model(data)  # Forward pass to the network

                # Compute loss based on output and ground truth
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3  # sum of losses
            # print("loss:", loss)

            # Compute gradients for each layer and update weights
            scaler.scale(loss).backward()  # backward pass: computes gradients
            scaler.step(optimizer)  # update weights based on accumulated gradients
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss="%.6f" % loss)
            step += 1
            writer.add_scalar("loss_step", loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar("epoch/loss_epoch_train", float(loss_train_mean), epoch)
        print("loss for train : %f" % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os

            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            # torch.save(model.state_dict(), f'{args.save_model_path}Saved_model_epoch_{epoch}.pth')
            torch.save(
                model.module.state_dict(),
                f"{args.save_model_path}Saved_model_epoch_{epoch}.pth",
            )

        if epoch % args.validation_step == 0 and epoch != args.num_epochs:
            precision, miou = val(args, model, dataloader_val)  # val() function call
            if miou > max_miou:
                max_miou = miou
                import os

                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(
                    model.module.state_dict(),
                    f"{args.save_model_path}Best_model_epoch_{epoch}.pth",
                )
            writer.add_scalar("epoch/precision_val", precision, epoch)
            writer.add_scalar("epoch/miou val", miou, epoch)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def parse_args(params):
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="train",
    )

    parse.add_argument(
        "--backbone",
        dest="backbone",
        type=str,
        default="CatmodelSmall",
    )
    parse.add_argument(
        "--pretrain_path",
        dest="pretrain_path",
        type=str,
        default="",
    )
    parse.add_argument(
        "--use_conv_last",
        dest="use_conv_last",
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        "--num_epochs", type=int, default=300, help="Number of epochs to train for"
    )
    parse.add_argument(
        "--epoch_start_i",
        type=int,
        default=0,
        help="Start counting epochs from this number",
    )
    parse.add_argument(
        "--checkpoint_step",
        type=int,
        default=10,
        help="How often to save checkpoints (epochs)",
    )
    parse.add_argument(
        "--validation_step",
        type=int,
        default=1,
        help="How often to perform validation (epochs)",
    )
    parse.add_argument(
        "--crop_height",
        type=int,
        default=512,
        help="Height of cropped/resized input image to modelwork",
    )
    parse.add_argument(
        "--crop_width",
        type=int,
        default=1024,
        help="Width of cropped/resized input image to modelwork",
    )
    parse.add_argument(
        "--batch_size", type=int, default=2, help="Number of images in each batch"
    )
    parse.add_argument(
        "--learning_rate", type=float, default=0.01, help="learning rate used for train"
    )
    parse.add_argument("--num_workers", type=int, default=4, help="num of workers")
    parse.add_argument(
        "--num_classes", type=int, default=19, help="num of object classes (with void)"
    )
    parse.add_argument(
        "--cuda", type=str, default="0", help="GPU ids used for training"
    )
    parse.add_argument(
        "--use_gpu", type=bool, default=True, help="whether to user gpu for training"
    )
    parse.add_argument(
        "--save_model_path", type=str, default=None, help="path to save model"
    )
    parse.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer, support rmsprop, sgd, adam",
    )
    parse.add_argument("--loss", type=str, default="crossentropy", help="loss function")

    return parse.parse_args()


def main(params):
    args = parse_args(params)

    ## dataset
    n_classes = args.num_classes
    mode = args.mode
    root = args.root

    train_dataset = CityScapes(mode)
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = CityScapes(mode="val")
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    ## model
    model = BiSeNet(
        backbone=args.backbone,
        n_classes=n_classes,
        pretrain_model=args.pretrain_path,
        use_conv_last=args.use_conv_last,
    )

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print("not supported optimizer \n")
        return None

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)


if __name__ == "__main__":
    params = [
        "--num_epochs",
        "50",
        "--learning_rate",
        "1e-2",
        "--pretrain_path",
        "/content/Drive/MyDrive/AML project/checkpoints/STDCNet813M_73.91.tar",
        "--path_dataset",
        "/content/Drive/MyDrive/Datasets/Cityscapes",
        "--num_workers",
        "8",
        "--num_classes",
        "19",
        "--cuda",
        "0",
        "--batch_size",
        "4",
        "--save_model_path",
        "/content/drive/MyDrive/checkpoints_101_sgd",
        "--context_path",
        "resnet101",  # set resnet18 or resnet101, only support resnet18 and resnet101
        "--optimizer",
        "sgd",
        "--loss",
        "crossentropy",
    ]

    main(params)
