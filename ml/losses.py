"""
Different loss functions. 
"""

import torch

def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon: float=1e-6):
    """ Calculate the dice loss between the predicted and target mask.

    Arguments
    ---------
        pred (torch.Tensor): the mask predicted by the machine learning model.
        target (torch.Tensor): the ground truth mask.
        epsilon (float): optional. Small value to prevent division by zero.

    Returns
    -------
        float: calculated dice loss
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return 1 - dice.mean()

