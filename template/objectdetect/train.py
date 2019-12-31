"""Trainning and validating."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: Dell, Tue Dec 31 17:08:42 CST 2019
# ***
# ************************************************************************************/
#

import math
import os
import sys
import argparse
import random
import torch
import torch.optim as optim

from data import get_data
from model import get_model, model_load, model_save
from coco_eval import create_coco_dataset, CocoEvaluator

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
    loss_box_reg = Average()
    loss_rpn_box = Average()
    loss_classifier = Average()
    loss_mask = Average()
    loss_objectness = Average()
    total_loss = Average()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # images, targets = images.to(device), targets.to(device)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
    model.eval()

    cpu_device = torch.device("cpu")

    coco = create_coco_dataset(loader.dataset)
    coco_evaluator = CocoEvaluator(coco, ["bbox", "segm"])  # keypoints

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Predict results without calculating gradients
            with torch.no_grad():
                outputs = model(images)

            # (Pdb) pp type(outputs), len(outputs), type(outputs[0]), outputs[0].keys()
            # (<class 'list'>,1,<class 'dict'>,dict_keys(['boxes', 'labels', 'scores', 'masks']))

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
    parser.add_argument('--outputdir', type=str, default="output", help="output directory")
    parser.add_argument('--checkpoint', type=str, default="output/maskrcnn.pth", help="checkpoint file")
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
