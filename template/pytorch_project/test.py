"""Model test."""

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
import os
import argparse
import torch
from data import get_data
from model import get_model, model_load, valid_epoch, model_setenv

if __name__ == "__main__":
    """Test model."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/{{ . }}.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # get model
    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')
