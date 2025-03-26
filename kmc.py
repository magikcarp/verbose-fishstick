#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Report efficacy of K-means clustering on Fluocell dataset using different
loss metrics. 

Usage:
    python kmc.py

It is expected that this file will be run from the parent directory of the
repository which includes a `data` directory holding the decompressed
Fluocell dataset images and masks. 
"""

import torch
import numpy as np
import cv2

from ml.datasets import FluoCellsDataset
from ml.losses import dice_loss

def predict_2mean(image: np.ndarray) -> np.ndarray:
    """ 
    Performs 2-mean clustering on provided grayscale image. 

    Args: 
        image (numpy.ndarray): image for analysis

    Returns:
        numpy.ndarray: segmented image
    """
    flat = image.reshape((-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.9)
    k = 2
    attempts = 10
    _, labels, centers = cv2.kmeans(
        flat,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    centers - np.float32(centers)
    flat_res = centers[labels.flatten()]
    res = flat_res.reshape((image.shape))
    return res

def report_loss_stats(loss_vals: list[float], loss_type: str) -> None:
    """
    Prints min, max, avg, and std of loss across a dataset.

    Args:
        loss_vals (list[float]): measured loss values for each image in dataset
        loss_type (str): name of method for calculating loss

    Returns:
        None
    """
    print(f"{loss_type} metrics")
    print(f" - min loss: {np.min(loss_vals)}")
    print(f" - max loss: {np.max(loss_vals)}")
    print(f" - avg loss: {np.average(loss_vals)}")
    print(f" - std loss: {np.std(loss_vals)}")

def main():
    # TODO for each image, calculate the dice loss
    # TODO report min, max, average, median, std dice loss across the dataset
    # TODO compare # of ROIs in pred vs true mask
    # TODO watershed & threshold size, compare to true mask
    fcds = FluoCellsDataset("data/")
    losses = [None] * len(fcds)
    for i, image in enumerate(fcds):
        img = np.array(image["image"][0, :, :].transpose(0, 1))
        msk = np.array(image["mask"][0, :, :].transpose(0, 1))
        msk_pred = predict_2mean(img)
        loss = dice_loss(torch.from_numpy(msk_pred), torch.from_numpy(msk))
        losses[i] = loss
        print(f"{i}: {loss}")
    report_loss_stats(losses, "Dice loss")
    return None

if __name__ == "__main__":
    main()
