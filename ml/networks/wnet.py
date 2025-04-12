#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" WNet constructed from UNet. """

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from .unet import UNet
from ..losses import NCutLoss

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

    def forward_enc(self, x: torch.Tensor) -> torch.Tensor:
        segmentation = self.enc_unet(x)
        return segmentation

    def forward_dec(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.dec_unet(x)
        return reconstruction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        segmentation = self.enc_unet(x)
        reconstruction = self.dec_unet(segmentation)
        return segmentation, reconstruction
    # TODO accept val set, return dict with epoch averaged losses
    def train_model(
            self,
            train_loader: torch.utils.data.DataLoader,
            n_epochs: int,
            lr: float=0.0005,
            verbose: bool=False,
            report_iter: int=25,
            ) -> None:
        self.train()

        optim_enc = torch.optim.Adam(self.enc_unet.parameters(), lr=lr)
        optim_all = torch.optim.Adam(self.parameters(), lr=lr)

        ncl = NCutLoss()
        
        for e in range(n_epochs):
            if verbose:
                print(f"Epoch {e+1}")
            for i, batch in enumerate(train_loader):
                # soft cut N loss pass
                optim_enc.zero_grad()
                seg = self.enc_unet(batch)
                soft_cut_loss = ncl(seg, batch)
                soft_cut_loss.backward(retain_graph=False)
                optim_enc.step()

                # complete forward pass
                optim_all.zero_grad()
                seg, recon = self(batch)
                recon_loss = mse_loss(seg, recon, reduction="sum")
                recon_loss.backward(retain_graph=False)
                optim_all.step()

                if verbose and i % report_iter == report_iter - 1:
                    print(f" - batch {i+1} loss: {recon_loss}")
