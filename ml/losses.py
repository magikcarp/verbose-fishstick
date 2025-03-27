# -*- coding: utf-8 -*-
"""
Different loss functions. 
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt


class DiceLoss:
    """ 
    Calculate the DICE loss between true and predicted mask. 

    DICE loss is defined as 2 * intersection / union of two images. 

    Attributes:
        e (float, optional): epsilon to prevent division by 0. Default is 1e-6. 
    """
    def __init__(self, epsilon: float=1e-6):
        """
        Initializes instance for calculating DICE loss.

        Args:
            e (float, optional): epsilon to prevent division by 0. 
                Default is 1e-6. 
        """
        self.e = epsilon

    def __call__(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor, 
                 ) -> torch.Tensor:
        """
        Calculate the dice loss between the predicted and target mask. 

        Predicted and true masks are expected to be of the same shape with four 
        dimensions [B x C x W x H] where B is the batch size, C is the number 
        of channels, W is the image width, and H is the image height. 

        Args:
            pred (torch.Tensor): mask predicted by the machine learning model
            target (torch.Tensor): ground truth mask

        Returns:
            float: calculated dice loss
        """
        # calculate dice coefficient
        intersection = (pred * target).sum(dim=pred.shape[-3:])
        union = pred.sum(dim=pred.shape[-3:]) + target.sum(dim=pred.shape[-3:])
        dice_coeff = (2. * intersection + self.e) / (union + self.e)

        # dice loss = 1 - dice coefficient
        return 1 - dice_coeff.mean()


class WeightedBCE:
    def __call__(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor
                 ) -> torch.Tensor:
        unweighted_bce = F.binary_cross_entropy(pred, target, reduction='none')


def build_pixel_penalty_map(
    y: torch.Tensor, 
    wc: dict=None, 
    w0: int=10, 
    sigma: int=25
) -> torch.Tensor:
    """ Creates a mask that emphasizes the space between nearby cells.

    Implementation of this was heavily inspired by StackOverflow discussion
    here: https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras/

    Args:
        mask (torch.Tensor): original mask in dataset.
        wc (dict): dictionary of class weights.
        w0 (int): border weight parameter. Default is 10. 
        sigma (int): border width parameter. Default is 25 as described by
            Morelli et al. in https://arxiv.org/abs/2103.01141.

    Returns:
        torch.Tensor: weight map
    """
    y_np = y.squeeze(1).detach().numpy()  # Convert to NumPy for processing
    batch_size, width, height = y_np.shape
    weight_maps = np.zeros((batch_size, width, height), dtype=np.float32)
    
    for b in range(batch_size):
        labels = label(y_np[b])
        no_labels = labels == 0
        label_ids = sorted(np.unique(labels))[1:]  # Exclude background (0)

        if len(label_ids) > 1:
            distances = np.zeros((width, height, len(label_ids)))
            for i, label_id in enumerate(label_ids):
                distances[:, :, i] = distance_transform_edt(labels != label_id)

            distances = np.sort(distances, axis=2)
            d1, d2 = distances[:, :, 0], distances[:, :, 1]
            w = w0 * np.exp(-0.5 * ((d1 + d2) / sigma) ** 2) * no_labels
        else:
            w = np.zeros_like(y_np[b])

        if wc:
            class_weights = np.zeros((width, height), dtype=np.float32)
            for k, v in wc.items():
                class_weights[y_np[b] == k] = v
            w += class_weights

        weight_maps[b] = w
    
    return torch.tensor(weight_maps, dtype=torch.float32).unsqueeze(1)
