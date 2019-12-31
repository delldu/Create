"""Data."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: {{ create "author" }}, {{ bash "date" }}
# ***
# ************************************************************************************/
#

import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T


class {{ . }}Dataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, transforms):
        """Init dataset."""
        super({{ . }}Dataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all images, sorting for alignment
        self.images = list(sorted(os.listdir(os.path.join(root, "Images"))))

    def __getitem__(self, idx):
        """Load images."""
        img_path = os.path.join(self.root, "Images", self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)


def get_transform(train):
    """Transform images."""
    ts = []
    if train:
        ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)


def get_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""
    train_ds = {{ . }}Dataset('images_root', get_transform(train=True))
    valid_ds = {{ . }}Dataset('images_root', get_transform(train=False))

    # Split train_ds in train and valid set
    indices = torch.randperm(len(train_ds)).tolist()
    train_ds = data.Subset(train_ds, indices[:-50])
    valid_ds = data.Subset(valid_ds, indices[-50:])

    # Define training and validation data loaders
    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(valid_ds, batch_size=bs * 2, shuffle=False, num_workers=4)

    return train_dl, valid_dl
