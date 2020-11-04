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
import glob
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model, model_load, model_setenv
from tqdm import tqdm

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
        from apex import amp
        model = amp.initialize(model, opt_level="O1")

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total = len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()

        # xxxx--modify here
        toimage(output_tensor.cpu()).show()
