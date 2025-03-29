#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a UNet model on the provided Fluocells dataset. 
"""

import torch
from torchvision import transforms

from ml.networks import UNet
import ml.datasets as ds

def main():
    the_works = transforms.Compose([
        ds.Flip(),
        ds.Crop(512),
        ds.Brighten(0.1),
        ds.Fuzz(),
        ds.Cap()
    ])
    fc_dsl = torch.utils.data.Dataloader(
        ds.FluoCellsDataset("data/", the_works),
        batch_size=8,
        shuffle=True,
    ) 
    unet = UNet()
    
    print(unet)

if __name__ == "__main__":
    main()
