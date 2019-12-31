"""Model predict."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright {{create "author"}} {{create "date +%Y"}}, All Rights Reserved.
# ***
# ***    File Author: {{ create "author"}}, {{bash "date" }}
# ***
# ************************************************************************************/
#

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model, model_load


if __name__ == "__main__":
    """Predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/{{ . }}.pth", help="checkpint file")
    parser.add_argument('--input', type=str, required=True, help="input image")
    args = parser.parse_args()

    # Running on GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    image = Image.open(args.input).convert("RGB")
    tensor = transforms.ToTensor()(image)
    input = tensor.to(device)

    # Predict without calculating gradients
    with torch.no_grad():
        output = model(input)

    # xxxx--modify here
    print(output)
