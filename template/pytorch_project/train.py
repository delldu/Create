"""Model trainning & validating."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright {{create "author"}} {{create "date +%Y"}}, All Rights Reserved.
# ***
# ***    File Author: {{ create "author" }}, {{ bash "date" }}
# ***
# ************************************************************************************/
#

import argparse
import os

import torch
import torch.optim as optim
from data import get_data
from model import get_model, model_device, model_save, train_epoch, valid_epoch

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str,
                        default="output", help="output directory")
    parser.add_argument('--checkpoint', type=str,
                        default="output/{{ . }}.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=8, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # get model
    model = get_model(args.checkpoint)
    device = model_device()
    model = model.to(device)

    # construct optimizer and learning rate scheduler,
    # xxxx--modify here
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.1)

    # get data loader
    train_dl, valid_dl = get_data(trainning=True, bs=args.bs)

    for epoch in range(args.epochs):
        print("Epoch {}/{}, learning rate: {} ...".format(epoch +
                                                          1, args.epochs, lr_scheduler.get_last_lr()))

        train_epoch(train_dl, model, optimizer, device, tag='train')

        valid_epoch(valid_dl, model, device, tag='valid')

        lr_scheduler.step()

        # xxxx--modify here
        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            model_save(model, os.path.join(
                args.outputdir, "latest-checkpoint.pth"))
