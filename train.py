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
            # print("\t",i)
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()  # Zero-ing the gradients
            # From: [batch_size, height, width, channels]
            # To: [batch_size, channels, height, width]
            # data = data.permute(0, 3, 1, 2)
            # print("input:",data.size())

            with amp.autocast():
                output, out16, out32 = model(data)  # Forward pass to the network
                # print("output:", output.size())
                # print("out16:", out16.size())
                # print("out32:", out32.size())
                # Compute loss based on output and ground truth
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3  # sum of losses
            # print("loss:", loss)

            # Compute gradients for each layer and update weights
            loss.backward(retain_graph=True)
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


def main(args, eval_only=False):

    n_classes = args.num_classes
    # mode = args.mode
    root = args.root

    if root == "Cityscapes/Cityspaces/":
        train_dataset = CityScapes(root, "train")
        val_dataset = CityScapes(root, "val")
    else:
        train_dataset = GTA5(root, "train")
        val_dataset = GTA5(root, "val")

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

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

    if args.epoch_start_i != 0:
        print(f"loading data from saved model {args.saved_model}")
        model.load_state_dict(torch.load(f"{args.save_model_path}{args.saved_model}"))

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

    if not eval_only:
        ## train loop
        train(args, model, optimizer, dataloader_train, dataloader_val)
    # final test
    val(args, model, dataloader_val)


class arguments:
    # mode = "train"
    backbone = "CatmodelSmall"
    pretrain_path = (
        "/content/Drive/MyDrive/Colab Notebooks/checkpoints/STDCNet813M_73.91.tar"
    )
    use_conv_last = False
    num_epochs = 50
    epoch_start_i = 0
    checkpoint_step = 5
    validation_step = 100
    crop_height = 512
    crop_width = 1024
    batch_size = 12
    learning_rate = 0.01
    num_workers = 2
    num_classes = 19
    cuda = "0"
    use_gpu = True
    save_model_path = "/content/Drive/MyDrive/Colab Notebooks/Partial models/"
    saved_model = f"Saved_model_epoch_{epoch_start_i}.pth"
    optimizer = "adam"
    loss = "crossentropy"
    root = "Cityscapes/Cityspaces/"
    # root='GTA5/'


if __name__ == "__main__":
    main_args = arguments()

    main(main_args, eval_only=False)
