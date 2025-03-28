# -*- coding: utf-8 -*-
"""
Different loss functions. 
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt


class Loss:
    pass


class DiceLoss(Loss):
    """ 
    Calculate the DICE loss between true and predicted mask. 

    DICE loss is defined as (2 * intersection / union) of two images. 

    Attributes:
        eps (float, optional): value prevent division by 0. Default is 1e-6. 
    """
    def __init__(self, epsilon: float=1e-6):
        """
        Initializes instance for calculating DICE loss.

        Args:
            eps (float, optional): epsilon to prevent division by 0. 
                Default is 1e-6. 
        """
        self.eps = epsilon

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
        intersection = (pred * target).sum(dim=(-1, -2, -3))
        union = pred.sum(dim=(-1, -2, -3)) + target.sum(dim=(-1, -2, -3))
        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)

        # dice loss = 1 - dice coefficient
        return 1 - dice_coeff.mean()


class WeightedBCE(Loss):
    """
    Calculate BCE loss using a weighted mask map. Produces reduced value. 

    Attributes:
        wc (dict, optional): an optional weight map of classes (0 as background 
            and 1 as the subject). 
        w0 (int, optional): weight of distance between cells. Default is 2. 
        sigma (int, optional): required proximity between cells. Default is 25. 
    """
    def __init__(self, 
                 wc: dict=None,
                 w0: int=2,
                 sigma: int=25
                 ):
        """
        Sets parameters required for calculating weighted BCE.

        Args: 
            wc (dict, optional)
            w0 (int, optional)
            sigma (int, optional)
        """
        self.wc = wc
        self.w0 = w0
        self.sigma = sigma

    def __call__(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor
                 ) -> torch.Tensor:
        """
        Calculates the weighted BCE between predicted and target masks.

        Args: 
            pred (torch.Tensor): predicted mask. 
            target (torch.Tensor): ground truth mask. 
        
        Returns: 
            torch.Tensor: computed BCE loss. 
        """
        unweighted_bce = F.binary_cross_entropy(pred, target, reduction='none')
        wm = build_penalty_map(target, wc=self.wc, w0=self.w0, sigma=self.sigma)
        weighted_bce = unweighted_bce * wm
        return weighted_bce.mean()


class SumLoss(Loss):
    """
    Sums the loss function means for a given prediction and target. 

    Attributes:
        losses (tuple[Loss]): collection of loss functions. 
    """
    def __init__(self, *loss_fns: Loss):
        """
        Initializes object for summing loss functions. 

        Args: 
            loss_fns: (tuple[Loss]): series of functions for total loss. 
        """
        self.losses = loss_fns

    def __call__(self, 
                 pred: torch.Tensor, 
                 target: torch.Tensor
                 ) -> torch.Tensor:
        """
        Calculates the summed loss for all previously included loss functions. 

        Args:
            pred (torch.Tensor): predicted mask. 
            target (torch.Tensor): ground truth mask. 

        Returns: 
            torch.Tensor: total calculated loss. 
        """
        out_loss = 0
        for loss in self.losses:
            out_loss += loss(pred, target)
        return out_loss
    

def build_penalty_map(
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
        torch.Tensor: weight map with same dimensions as input mask. 
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
