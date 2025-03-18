import os

import torch
from torchvision import transforms
import numpy as np
import cv2

import matplotlib.pyplot as plt
from random import randint


class ImageSample:
    """ Gray-scale image and mask with dimensions [C x W x H] where C is the
        number of color channels, W is the width of the image, and H is the
        height of the image. 
    """
    def __init__(self, image: torch.Tensor, mask: torch.Tensor):
        self.image = image
        self.mask = mask

    def display(self):
        """ Displays the image and mask using matplotlib, primarily useful for 
            Jupyter notebooks. 
        """
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(self.image.transpose(0, 2))
        plt.subplot(122)
        plt.title("Mask")
        plt.imshow(self.mask.transpose(0, 2))
        

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


class Flip:
    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample.image, sample.mask
        h_flip = randint(0, 1)
        v_flip = randint(0, 1)

        if h_flip:
            print("H FLIP!")
            img = torch.flip(img, [1])
            msk = torch.flip(msk, [1])
        if v_flip:
            print("V FLIP!")
            img = torch.flip(img, [2])
            msk = torch.flip(msk, [2])
            
        return ImageSample(img, msk)


class Crop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample.image, sample.mask
        max_w = img.size()[1] - self.size - 1
        max_h = img.size()[2] - self.size - 1

        w = randint(0, max_w)
        h = randint(0, max_h)
        print(f"NEW CORNER: {w}, {h}")
        
        new_img = img[:, w:w + self.size, h:h + self.size]
        new_msk = msk[:, w:w + self.size, h:h + self.size]

        return ImageSample(new_img, new_msk)

class Brighten:
    """ Brightens (or darkens) an image by a random constant value. 


    """
    def __init__(self, max_bound: float):
        self.max = max_bound

    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample.image, sample.mask
        bright_adjust = (torch.rand(1) - 0.5) * self.max
        print(f"BRIGHT ADJUST: {bright_adjust}")
        img = img + bright_adjust 
        return ImageSample(img, msk)


class Fuzz:
    """ Adds random noise to the image using different distributions. Images
        can add Gaussian, uniform, or no noise which is determined randomly. 
    """
    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample.image, sample.mask
        noise = randint(0, 2)

        if noise == 2:
            gaussian_noise = torch.randn(sample.image.shape) * 0.1
            print("GAUSSIAN")
            img = img + gaussian_noise
        elif noise == 1:
            uniform_noise = (torch.rand(sample.image.shape) - 0.5) / 5
            print("UNIFORM")
            img = img + uniform_noise
        else:
            print("NO NOISE!")
            pass # no noise added

        return ImageSample(img, msk)


class Cap:
    """ Ensures values within the image do not exceed the expected bounds of
        float values on the interval [0, 1]. This should be the last
        transformation included in the series before training or predictions. 
    """
    def __call__(self, sample: ImageSample) -> ImageSample:
        img = sample.image
        img[img > 1] = 1
        img[img < 0] = 0
        return ImageSample(img, sample.mask)

