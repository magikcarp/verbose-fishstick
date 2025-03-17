import os

import torch
from torchvision import transforms
import numpy as np
import cv2

import matplotlib.pyplot as plt


class ImageSample:
    """ Gray-scale image and mask with dimensions [C x W x H] where C is the
        number of color channels, W is the width of the image, and H is the
        height of the image. 
    """
    def __init__(self, image: torch.Tensor, mask: torch.Tensor):
        self.image = image
        self.mask = mask

    def display(self):
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(self.image.transpose(0, 2))
        plt.subplot(122)
        plt.title("Mask")
        plt.imshow(self.mask.transpose(0, 2))
        pass


class FluoCellsDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms: transforms.Compose=None):
        self.root = root
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, index: int) -> ImageSample:
        img_path = os.path.join(self.root, "images", self.images[index])
        msk_path = os.path.join(self.root, "masks", self.masks[index])

        img = torch.from_numpy(
            np.expand_dims(
                cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY),
                axis=0
            )
        ).float().transpose(1, 2) / 255
        msk = torch.from_numpy(
            np.expand_dims(
                cv2.cvtColor(cv2.imread(msk_path), cv2.COLOR_RGB2GRAY),
                axis=0
            )
        ).float().transpose(1, 2) / 255
        
        
        smpl = ImageSample(img, msk)

        if self.transforms:
            smpl = self.transforms(smpl)

        return smpl

    def __len__(self):
        return len(self.images)


class Tile:
    def __init__(self, out_size: int):
        self.out_size = out_size

    def __call__(self, sample: ImageSample):
        pass
        

