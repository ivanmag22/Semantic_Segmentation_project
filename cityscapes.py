#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
import os
import os.path
from pathlib import Path
from transforms import *
import numpy as np
from PIL import Image


class CityScapes(Dataset):
    def __init__(self, base_root, mode):
        super(CityScapes, self).__init__()

        self.mode = mode
        self.image_paths = []  # images
        self.labels = []  # labels

        assert mode == "train" or mode == "val"  # just checking for potential issues
        image_folder = f"{base_root}images/{mode}"

        for root, dirs, files in os.walk(image_folder):
            for file_name in files:
                image_path = f"{root}/{file_name}"
                assert Path(image_path).is_file()
                self.image_paths.append(image_path)

                label = image_path.replace("leftImg8bit", "gtFine_labelTrainIds")
                label = label.replace("/images/", "/gtFine/")
                assert Path(label).is_file()
                self.labels.append(label)

        if self.mode == "train":
            self.transform = train_transform
        if self.mode == "val":
            self.transform = eval_transform
        self.label_transform = label_transform

        assert len(self.image_paths) != 0
        assert len(self.image_paths) == len(self.mask_paths_bw)

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index]).convert("RGB")
        label = Image.open(self.mask_paths_bw[index])

        if self.mode == "train":
            label = np.array(self.label_transform(label))[np.newaxis, :]
        else:
            label = np.array(label)[np.newaxis, :]

        image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_paths)
