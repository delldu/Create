"""Trainning/validating."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: {{ create "author" }}, {{ bash "date" }}
# ***
# ************************************************************************************/
#

import math
import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_data
from model import get_model, model_load, model_save

from tqdm import tqdm


class Average(object):
    """Class Average."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Average()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.4f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_one_epoch(loader, model, device, tag=''):
    """Validating model  ..."""
    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            t.update(count)


if __name__ == "__main__":
    """Trainning model."""
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str, default="output", help="output directory")
    parser.add_argument('--checkpoint', type=str, default="output/{{ . }}.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test', type=bool, default=False, help="test model")
    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # train on the GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get model
    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)

    if args.test:
        print("Start testing ...")
        #     test_dl = None
        #     valid_one_epoch(test_dl, model, device, tag='test')
        sys.exit(0)

    # construct optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # get data loader
    train_dl, valid_dl = get_data(args.bs)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {} ...".format(epoch + 1, args.epochs, lr_scheduler.get_lr()))

        train_one_epoch(train_dl, model, optimizer, device, tag='train')

        valid_one_epoch(valid_dl, model, device, tag='valid')

        lr_scheduler.step()

        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            model_save(model, os.path.join(args.outputdir, "latest-checkpoint.pth"))
