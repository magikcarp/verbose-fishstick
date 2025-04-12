import os

import torch
from torchvision import transforms
import numpy as np
import cv2

import matplotlib.pyplot as plt
from random import randint


class ImageSample(dict):
    """ 
    Gray-scale image and mask to be used for machine learning tasks. 

    Gray-scale image and mask with dimensions [C x W x H] where C is the
    number of color channels, W is the width of the image, and H is the
    height of the image. Pixels in the image are represented by float values.
    Values in the image should lie on the open range [0, 1] and values in the
    mask should be either 0 or 1. There are no internal checks to validate
    these assumptions.

    Attributes (stored in dictionary format for DataLoader compatibility):
        "image": torch.Tensor of the sample image
        "mask": torch.Tensor of the mask to the corresponding image

    Methods
        display: shows image and mask using matplotlib
    """
    def __init__(self, image: torch.Tensor, mask: torch.Tensor):
        """ Initializes with desired image and mask.

        Arguments:
            image (torch.Tensor): image of sample
            mask (torch.Tensor): mask associated with image
        """
        self["image"] = image
        self["mask"] = mask

    def display(self):
        """ Displays the image and mask using matplotlib, primarily useful for 
            Jupyter notebooks. 
        """
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(self["image"].transpose(0, 2))
        plt.subplot(122)
        plt.title("Mask")
        plt.imshow(self["mask"].transpose(0, 2))
        

class FluoCellsDataset(torch.utils.data.Dataset):
    """ 
    Fluocells dataset.

    Attributes:
        images: list of image names in image directory
        masks: list of mask names in masks dataset
        root: string representation of directory path to dataset
        transforms: torchvision.transforms.Compose
    """
    def __init__(self, root: str, transforms: transforms.Compose=None):
        """ 
        Initializes dataset to read samples.

        Args:
            root (str): the relative or absolute path to the parent directory
                of the dataset
            transforms (torchvision.transforms.Compose): series of
                transformations of the data that occur in sequential order
        """
        self.root = root
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, index: int) -> ImageSample:
        """ 
        Retreives the nth image and mask in alphabetical order.

        Image and mask names are retreived from the `images` and `masks`
        attributes, respectively. Images are read using opencv-python,
        converted to a grayscale image, converted to a torch.Tensor, and
        converted from an integer to a float value representation.  
        
        Args:
            index (int): the nth image and mask

        Returns:
            ImageSample: image and mask each encoded in a torch.Tensor    
        """
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
        """ Provides the number of samples in the dataset. """
        return len(self.images)


class DapiTifDataset(torch.utils.data.Dataset):
    """ 
    Dataset for TIF images to be predicted using neural network. 

    Attributes:
        images: list of TIF DAPI images for prediction
        root: directory housing the DAPI images
    """
    def __init__(self, root: str):
        self.images = list(sorted(os.listdir(os.path.join(root, "DAPI"))))
        self.root = root
        
    def __getitem__(self, index: int) -> torch.Tensor:
        # TIF color channels max at 64
        # https://stackoverflow.com/questions/33610825/normalization-in-image-processing
        img_path = os.path.join(self.root, "DAPI", self.images[index])
        img = cv2.imread(img_path)
        img = torch.from_numpy(
            np.expand_dims(img[:, :, 0], axis=0)
        ).float().transpose(1, 2) / 64
        return img

    def __len__(self):
        return len(self.images)


class Flip:
    """ Randomly flips image and mask vertically and/or horizontally. """
    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample["image"], sample["mask"]
        h_flip = randint(0, 1)
        v_flip = randint(0, 1)

        if h_flip:
            # print("H FLIP!")
            img = torch.flip(img, [1])
            msk = torch.flip(msk, [1])
        if v_flip:
            # print("V FLIP!")
            img = torch.flip(img, [2])
            msk = torch.flip(msk, [2])
            
        return ImageSample(img, msk)


class Crop:
    """ Obtains random crop of the image and mask. """
    def __init__(self, size: int):
        self.size = size

    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample["image"], sample["mask"]
        max_w = img.size()[1] - self.size - 1
        max_h = img.size()[2] - self.size - 1

        w = randint(0, max_w)
        h = randint(0, max_h)
        # print(f"NEW CORNER: {w}, {h}")
        
        new_img = img[:, w:w + self.size, h:h + self.size]
        new_msk = msk[:, w:w + self.size, h:h + self.size]

        return ImageSample(new_img, new_msk)

class Brighten:
    """ Brightens image +- a random constant value. """
    def __init__(self, max_bound: float):
        self.max = max_bound

    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample["image"], sample["mask"]
        bright_adjust = ((torch.rand(1) * 2) - 1) * self.max
        # print(f"BRIGHT ADJUST: {bright_adjust}")
        img = img + bright_adjust 
        return ImageSample(img, msk)


class Fuzz:
    """ Adds random noise from the Gaussian, uniform, or no distribution. """
    def __call__(self, sample: ImageSample) -> ImageSample:
        img, msk = sample["image"], sample["mask"]
        noise = randint(0, 2)

        if noise == 2:
            gaussian_noise = torch.randn(sample["image"].shape) * 0.1
            # print("GAUSSIAN")
            img = img + gaussian_noise
        elif noise == 1:
            uniform_noise = (torch.rand(sample["image"].shape) - 0.5) / 5
            # print("UNIFORM")
            img = img + uniform_noise
        else:
            # print("NO NOISE!")
            pass # no noise added

        return ImageSample(img, msk)


class Cap:
    """ Caps float values in image to predefined open range of [0, 1]. """
    def __call__(self, sample: ImageSample) -> ImageSample:
        img = sample["image"]
        # print(f"PRE-CAP MIN {torch.min(img):.2f} MAX {torch.max(img):.2f}")
        img[img > 1] = 1
        img[img < 0] = 0
        # print(f"POST-CAP MIN {torch.min(img):.2f} MAX {torch.max(img):.2f}")
        return ImageSample(img, sample["mask"])
