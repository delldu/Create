
import pdb
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model_instance_segmentation, model_load
from utils import blend_image


if __name__ == "__main__":
    """Predict."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="output/maskrcnn.pth")
    parser.add_argument('--input_image', type=str, required=True)
    args = parser.parse_args()

    # Running on the GPU if a GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cpu_device = torch.device("cpu")

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model_load(model, args.checkpoint)

    # move model to the right device
    model.to(device)
    model.eval()

    image = Image.open(args.input_image).convert("RGB")
    tensor = transforms.ToTensor()(image)

    images = [tensor.to(device)]
    # Predict results without calculating gradients
    with torch.no_grad():
        outputs = model(images)

    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

    # pdb.set_trace()
    # (Pdb) pp type(outputs), len(outputs), type(outputs[0]), outputs[0].keys()
    # (<class 'list'>,
    #  1,
    #  <class 'dict'>,
    #  dict_keys(['boxes', 'labels', 'scores', 'masks']))
    scores = outputs[0]['scores'].cpu().tolist()
    scores = [s for s in scores if s > 0.60]
    count = len(scores)
    if count > 0:
        boxes = outputs[0]['boxes'].cpu()
        labels = outputs[0]['labels'].cpu()
        masks = outputs[0]['masks'].cpu()
        boxes = boxes[0:count]
        labels = labels[0:count]
        masks = masks[0:count]
        result = blend_image(image, labels, boxes, masks, scores)
        result.show()
    else:
        image.show()

