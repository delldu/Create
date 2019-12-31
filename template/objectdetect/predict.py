"""Predict."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: Dell, Tue Dec 31 17:08:42 CST 2019
# ***
# ************************************************************************************/
#

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model, model_load
from utils import blend_image

if __name__ == "__main__":
    """Predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/maskrcnn.pth", help="checkpint file")
    parser.add_argument('--input', type=str, required=True, help="input image")
    args = parser.parse_args()

    # Running on GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cpu_device = torch.device("cpu")

    model = get_model()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    image = Image.open(args.input).convert("RGB")
    tensor = transforms.ToTensor()(image)
    inputs = [tensor.to(device)]

    # Predict without calculating gradients
    with torch.no_grad():
        outputs = model(inputs)

    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

    # (Pdb) pp type(outputs), len(outputs), type(outputs[0]), outputs[0].keys()
    # (<class 'list'>, 1,  <class 'dict'>, dict_keys(['boxes', 'labels', 'scores', 'masks']))
    scores = outputs[0]['scores'].tolist()
    scores = [s for s in scores if s > 0.60]
    count = len(scores)
    if count > 0:
        boxes = outputs[0]['boxes'][0:count]
        labels = outputs[0]['labels'][0:count]
        masks = outputs[0]['masks'][0:count]
        result = blend_image(image, labels, boxes, masks, scores)
        result.show()
    else:
        image.show()

