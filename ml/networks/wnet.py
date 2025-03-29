#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" WNet constructed from UNet. """

import torch
import torch.nn as nn

from .unet import UNet


class WNet(nn.Module):
    """
    W-Net architecture for unsupervised image segmentation. 

    Inspired by the description as outlined in https://arxiv.org/abs/1711.08506.

    Attributes:
        enc_unet (UNet): UNet arm that encodes images
        dec_unet (UNet): UNet arm that decodes image representation

    The number of kernels can be customized if desired. Example network sizes:
    - UNet : [64, 128, 256, 512, 1024]
    - SmallUNet : [32, 64, 128, 256, 512]
    - MiniUNet : [16, 32, 64, 128, 256]
    - TinyUNet : [8, 16, 32, 64, 128]
    - DeepUNet : [8, 16, 32, 64, 128, 256, 512, 1024]
        - Should only be used with very large images due to down sampling
        (image demnsions are halved after each layer of double convolution)
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 unet_shape: list[int]):
        super().__init__()
        self.enc_unet = UNet(in_channels, out_channels, unet_shape)
        self.dec_unet = UNet(out_channels, in_channels, unet_shape)

        def forward_enc():
            pass

        def forward_dec():
            pass

        def forward():
            pass
