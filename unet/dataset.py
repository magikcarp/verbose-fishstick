import os

import torch
from torchvision import transforms
import numpy as np
import cv2


class ImageSample:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask


class FluoCellsDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms: transforms.Compose=None):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, index: int) -> ImageSample:
        img_path = os.path.join(self.root, "images", self.imgs[index])
        mask_path = os.path.join(self.root, "images", self.masks[index])

        img = cv2.imread(img_path)
        msk = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        sample = {"image": img, "mask": msk}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class Resize:
    def __init__(self, out_size: int):
        self.out_size = out_size

    def __call__(self, sample: ImageSample) -> ImageSample:
        img_resized = cv2.resize(sample.image, (self.out_size, self.out_size))
        msk_resized = torch.from_numpy(sample.image).float().transpose(0, 2) / 255
        return ImageSample(img_resized, msk_resized)


class ToTensor:
    def __call__(self, sample: ImageSample):
        img_tensor = torch.from_numpy(sample.image).float().transpose(0, 2) / 255
        msk_tensor = torch.from_numpy(
            np.expand_dims(sample.mask, axis=0)
        ).float().transpose(1, 2)
        return ImageSample(img_tensor, msk_tensor)
