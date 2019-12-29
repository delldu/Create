"""
Sample code from the TorchVision Object Detection Finetuning Tutorial.

http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""

import pdb
import math
import os
import sys
import argparse
import random
import torch
import torchvision.transforms as T

from dataset import PennFudanDataset
from model import get_model_instance_segmentation, get_iou_type, model_load
from coco_eval import create_coco_dataset_from_ourdataset, CocoEvaluator

from tqdm import tqdm


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)


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


def update_lr(optimizer, epoch):
    """Update learning rate."""
    start_learning_rate = 1e-4
    learning_gamma = 0.01
    learning_steps = 100

    current_lr = start_learning_rate * (learning_gamma ** ((epoch + 1) // learning_steps))
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr
    return current_lr


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


def train_one_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""
    loss_box_reg = Average()
    loss_rpn_box = Average()
    loss_classifier = Average()
    loss_mask = Average()
    loss_objectness = Average()
    total_loss = Average()

    # Set the model to training mode
    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # images, targets = images.to(device), targets.to(device)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            torch.cuda.synchronize()

            loss_dict = model(images, targets)
            # pdb.set_trace()
            # (Pdb) pp loss_dict
            # {'loss_box_reg': tensor(0.2717, device='cuda:0', grad_fn=<DivBackward0>),
            #  'loss_classifier': tensor(0.7600, device='cuda:0', grad_fn=<NllLossBackward>),
            #  'loss_mask': tensor(3.8534, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
            #  'loss_objectness': tensor(0.0197, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
            #  'loss_rpn_box_reg': tensor(0.0239, device='cuda:0', grad_fn=<DivBackward0>)}

            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            loss_box_reg.update(loss_dict['loss_box_reg'].item(), count)
            loss_rpn_box.update(loss_dict['loss_rpn_box_reg'].item(), count)
            loss_classifier.update(loss_dict['loss_classifier'].item(), count)
            loss_mask.update(loss_dict['loss_mask'].item(), count)
            loss_objectness.update(loss_dict['loss_objectness'].item(), count)
            total_loss.update(loss_value, count)

            t.set_postfix(
                loss='{:.4f},box_reg:{:.4f},rpn_box:{:.4f},class:{:.4f},mask:{:.4f},object:{:.4f}'
                .format(total_loss.avg, loss_box_reg.avg, loss_rpn_box.avg,
                        loss_classifier.avg, loss_mask.avg, loss_objectness.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        return total_loss.avg


def valid_one_epoch(loader, model, device, tag=''):
    """Validating model  ..."""
    # Set the model to evaluation mode
    model.eval()

    cpu_device = torch.device("cpu")

    coco = create_coco_dataset_from_ourdataset(loader.dataset)
    coco_evaluator = CocoEvaluator(coco, get_iou_type(model))

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            torch.cuda.synchronize()

            # Predict results without calculating gradients
            with torch.no_grad():
                outputs = model(images)

            # pdb.set_trace()
            # (Pdb) pp type(outputs), len(outputs), type(outputs[0]), outputs[0].keys()
            # (<class 'list'>,
            #  1,
            #  <class 'dict'>,
            #  dict_keys(['boxes', 'labels', 'scores', 'masks']))

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            res = {
                target["image_id"].item(): output
                for target, output in zip(targets, outputs)
            }
            coco_evaluator.update(res)

            t.update(count)

        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()


if __name__ == "__main__":
    """Trainning model."""
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default="output")
    parser.add_argument('--checkpoint', type=str, default="maskrcnn.pth")
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    # Create a directory to store weights
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # train on the GPU if a GPU is available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model_load(model, os.path.join(args.output_dir, args.checkpoint))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    for epoch in range(args.epochs):
        train_one_epoch(data_loader,
                        model,
                        optimizer,
                        device,
                        tag='[train {}/{}]'.format(epoch + 1, args.epochs))

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        if epoch % 2 == 0:
            valid_one_epoch(data_loader_test, model, device, tag='valid')

        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            model_save(model, os.path.join(args.output_dir, args.checkpoint))
