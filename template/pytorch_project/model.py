"""Create model."""

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
import sys
import math
import torch
import torch.nn as nn
from tqdm import tqdm

class {{ . }}Model(nn.Module):
    """{{ . }} Model."""

    def __init__(self):
        """Init model."""
        super({{ . }}Model, self).__init__()

    def forward(self, x):
        """Forward."""
        return x

def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)

def model_export():
    """Export model to onnx."""

    import onnx
    from onnx import optimizer

    # xxxx--modify here
    onnx_file = "model.onnx"
    weight_file = "checkpoint/weight.pth"

    # 1. Load model
    print("Loading model ...")
    model = {{ . }}Model()
    model_load(model, weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    # xxxx--modify here
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(model, dummy_input, onnx_file,
                    input_names=input_names, 
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    keep_initializers_as_inputs=True,
                    export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"


def get_model():
    """Create model."""
    model = {{ . }}Model()
    return model


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


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Average()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # xxxx--modify here
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.4f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            if os.environ["ENABLE_APEX"] == "YES":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Average()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            # xxxx--modify here
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.4f}'.format(valid_loss.avg))
            t.update(count)


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ["ONLY_USE_CPU"] == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else: # GPU ?
        os.environ["ENABLE_APEX"] = "YES"
        try:
            from apex import amp
        except:
            os.environ["ENABLE_APEX"] = "NO"

    # Running on GPU if available
    if os.environ["ONLY_USE_CPU"] == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Environment")
    print("----------------------------------------------")
    print("  USER: ", os.environ["USER"])
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVCIE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def infer_perform():
    """Model infer performance ..."""

    model_setenv()
    device = os.environ["DEVICE"]

    model = {{ . }}Model()
    model.eval()
    model = model.to(device)

    with tqdm(total=len(1000)) as t:
        t.set_description(tag)

        # xxxx--modify here
        input = torch.randn(64, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        t.update(1)


if __name__ == '__main__':
    model_export()
    infer_perform()
