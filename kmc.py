#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Report results of K-means clustering on Fluocell dataset.

usage:
    python kmc.py
"""

import torch
import numpy as np
import cv2

from ml.datasets import FluoCellsDataset
from ml.losses import dice_loss

def predict_2mean(image: np.ndarray) -> np.ndarray:
    flat = image.reshape((-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.9)
    k = 2
    attempts = 10
    ret, labels, centers = cv2.kmeans(
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
    print("METRICS")
    print(f" - MIN dice loss: {np.min(losses)}")
    print(f" - MAX dice loss: {np.max(losses)}")
    print(f" - AVG dice loss: {np.average(losses)}")
    print(f" - STD dice loss: {np.std(losses)}")
    return None

if __name__ == "__main__":
    main()
