"""Onnx Model Tools."""
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
import os
import pdb  # For debug
import time

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision.transforms as transforms
from PIL import Image

#
# /************************************************************************************
# ***
# ***    MS: Import Model Method
# ***
# ************************************************************************************/
#
from model import get_model

def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print("Onnx Model Engine: ", onnx_model.get_providers(),
          "Device: ", onnxruntime.get_device())

    return onnx_model


def onnx_forward(onnx_model, input):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])


if __name__ == '__main__':
    """Onnx tools ..."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--export', help="export onnx model", action='store_true')
    parser.add_argument('-v', '--verify', help="verify onnx model", action='store_true')
    parser.add_argument('-p', '--predict', help="predict with onnx model", action='store_true')
    parser.add_argument('-o', '--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #
    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    #

    dummy_input = torch.randn(1, 1, 1, 512)
    onnx_file_name = "{}/model.onnx".format(args.output)

    def export_onnx():
        """Export onnx model."""

        # 1. Create and load model.
        torch_model = get_model()
        torch_model = torch_model.cuda()
        torch_model.eval()

        # 2. Model export
        print("Exporting onnx model to {}...".format(onnx_file_name))

        input_names = ["input"]
        output_names = ["output"]
        # dynamic_axes = {'input': {0: "batch"},'output': {0: "batch"}}

        torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                          input_names=input_names,
                          output_names=output_names,
                          verbose=True,
                          opset_version=11,
                          keep_initializers_as_inputs=False,
                          export_params=True)

        # 3. Optimize model
        # print('Checking model ...')
        # onnx_model = onnx.load(onnx_file_name)
        # onnx.checker.check_model(onnx_model)
        # https://github.com/onnx/optimizer

        # 4. Visual model
        # python -c "import netron; netron.start('output/model.onnx')"

    def verify_onnx():
        """Verify onnx model."""

        torch_model = get_model()
        torch_model.eval()

        onnxruntime_engine = onnx_load(onnx_file_name)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        with torch.no_grad():
            torch_output = torch_model(dummy_input)

        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03)
        print("Onnx model {} has been tested with ONNXRuntime, result sounds good !".format(onnx_file_name))

    def onnx_predict():
        print("Onnx predicting ...")

        #
        # /************************************************************************************
        # ***
        # ***    MS: Define Input/Output File
        # ***
        # ************************************************************************************/
        #

        input_image_file = "lena.png"
        output_image_file = "{}/{}".format(args.output, input_image_file)

        # /************************************************************************************
        # ***
        # ***    MS: Normal Predict Flow
        # ***
        # ************************************************************************************/
        image = Image.open(input_image_file).convert("RGB")
        onnx_model = onnx_load(onnx_file_name)

        start_time = time.time()
        totensor = transforms.ToTensor()
        toimage = transforms.ToPILImage()

        input_image_tensor = totensor(input_image).unsqueeze(0)
        output_image_tensor = onnx_forward(onnx_model, input_image_tensor)
        output_image = toimage(output_image_tensor.squeeze(0))

        spend_time = time.time() - start_time
        print("Spend time: {:.2f} seconds".format(spend_time))

        output_image.save(output_image_file)

    #
    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    #

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()

    if args.predict:
        onnx_predict()
