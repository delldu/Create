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

import argparse
import torch
from data import get_data
from model import get_model, model_load, valid_epoch


if __name__ == "__main__":
    """Test model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/{{ . }}.pth", help="checkpoint file")
    args = parser.parse_args()

    # Test on GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get model
    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)

    print("Start testing ...")
    # xxxx--modify here
    #     test_dl = None
    #     valid_epoch(test_dl, model, device, tag='test')
