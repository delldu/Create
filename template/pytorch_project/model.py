"""Model."""

# coding=utf-8
#
# /************************************************************************************
# ***
# ***    File Author: {{ create "author" }}, {{ bash "date" }}
# ***
# ************************************************************************************/
#

import os
import torch


def model_load(model, path):
    """Load model."""
    if not os.path.exists(path):
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


def get_model():
    """Create model."""
    model = {{ . }}Model()

    return model
