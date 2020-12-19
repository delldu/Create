"""{{.}} appliction class."""
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

import inspect
import os

import torch.nn as nn

from .model import get_model


class {{.}}(nn.Module):
    """{{ . }}."""

    def __init__(self, weight_file="{{ . }}.pth"):
        """Init model."""
        super({{.}}, self).__init__()
        checkpoint = os.path.join(inspect.getfile(
            self.__init__), '..', 'weights/%s' % (weight_file))
        model, device = get_model(checkpoint)

        self.model = model
        self.device = device

    def forward(self, x):
        """Forward."""

        return self.model(x)
