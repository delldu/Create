"""Data."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: Dell, Tue Dec 31 17:08:42 CST 2019
# ***
# ************************************************************************************/
#

import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data

import torchvision.transforms as T


class PennFudanDataset(data.Dataset):
    """Define train_ds."""

    def __init__(self, root, transforms):
        """Init train_ds."""
        super(PennFudanDataset, self).__init__()

        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.images = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        """Load images ad masks."""
        img_path = os.path.join(self.root, "PNGImages", self.images[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)


def collate_fn(batch):
    """Collate fn."""
    return tuple(zip(*batch))


def get_transform(train):
    """Transform images."""
    ts = []
    if train:
        ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)


def get_data(bs):
    """Get data loader for trainning and validating, bs -- batch_size ."""
    train_ds = PennFudanDataset('PennFudanPed', get_transform(train=True))
    valid_ds = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the train_ds in train and test set
    indices = torch.randperm(len(train_ds)).tolist()
    train_ds = data.Subset(train_ds, indices[:-50])
    valid_ds = data.Subset(valid_ds, indices[-50:])

    # define training and validation data loaders
    train_dl = data.DataLoader(train_ds,
                               batch_size=bs,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_fn)

    valid_dl = data.DataLoader(valid_ds,
                               batch_size=bs * 2,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=collate_fn)

    return train_dl, valid_dl
