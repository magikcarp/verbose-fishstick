#!/usr/bin/env python

""" WNet constructed from UNet. """

import torch

from .unet import UNet, MiniUNet


class WNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
