from torch.utils.data import Subset, DataLoader, Dataset
import os
import os.path
from pathlib import Path
from transforms import *
from augmentation import augmentation_transforms
import json
import random
from PIL import Image


class GTA5(Dataset):
    def __init__(self, base_root, mode, augmentation=False, train_test_rateo=2 / 3):
        super(GTA5, self).__init__()

        self.mode = mode
        self.image_paths = []  # images
        self.label_paths = []  # labels
        self.train_test_rateo = train_test_rateo
        self.augmentation = augmentation
        with open("STDC_seg/cityscapes_info.json", "r") as fr:
            labels_info = json.load(fr)
        self.label_map = {el["id"]: el["trainId"] for el in labels_info}
        self.label_map.update({34: 255})

        assert mode == "train" or mode == "val"  # just checking for potential issues

        image_folder = f"{base_root}images"

        for root, dirs, files in os.walk(image_folder):
            for file_name in files:
                image_path = f"{root}/{file_name}"
                assert Path(image_path).is_file()
                self.image_paths.append(image_path)

                label_path = image_path.replace("/images/", "/labels/")
                assert Path(label_path).is_file()
                self.label_paths.append(label_path)

        l = int(len(self.image_paths) * self.train_test_rateo)
        if self.mode == "train":
            self.image_paths = self.image_paths[:l]
            self.label_paths = self.label_paths[:l]
            self.image_transform = gta_train_transform
            self.label_transform = gta_label_transform
        elif self.mode == "val":
            self.image_paths = self.image_paths[l:]
            self.label_paths = self.label_paths[l:]
            self.image_transform = gta_val_transform

        assert len(self.image_paths) != 0
        assert len(self.image_paths) == len(self.label_paths)

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")
        image = self.image_transform(image)
        aug_ind = random.choice(range(len(augmentation_transforms)))
        if self.augmentation and random.choice([True, False]):
            # aug_transform = transforms.Compose(random.sample(augmentation_transforms, 2))
            # image = aug_transform(image)
            image = augmentation_transforms[aug_ind](image)

        label = Image.open(self.label_paths[index])
        if self.mode == "train":
            label = self.label_transform(label)
            if (
                aug_ind == 5 or aug_ind == 6
            ):  # spacial transformation must be applied to the label aswell
                label = augmentation_transforms[aug_ind](label)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)

        return image, label

    def convert_labels(self, label):
        for k, v in self.label_map.items():
            label[label == k] = v
        return label

    def __len__(self):
        return len(self.image_paths)
