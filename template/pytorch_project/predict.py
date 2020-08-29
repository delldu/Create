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
import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model, model_load, model_setenv

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/{{ . }}.pth", help="checkpint file")
    parser.add_argument('--input', type=str, required=True, help="input image")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    if os.environ["ENABLE_APEX"] == "YES":
        model, = amp.initialize(model, opt_level="O1")

    image = Image.open(args.input).convert("RGB")
    tensor = transforms.ToTensor()(image)
    input = tensor.to(device)

    # Predict without calculating gradients
    with torch.no_grad():
        output = model(input)

    # xxxx--modify here
    print(output)
