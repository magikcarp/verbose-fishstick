#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Establishes architecture for UNet model. """


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable


class UNet(nn.Module):
    """ UNet network that produces a mask. 
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Encoding arm
        self.down_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in
              [(in_channels, 64), (64, 128), (128, 256), (256, 512)]]
        )
        self.down_sample = nn.ModuleList(
            [DownSample() for _ in range(4)]
        )

        # Bottom of the U
        self.middle_conv = DoubleConvolution(512, 1024)

        # Decoding arm
        self.up_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in
              [(1024, 512), (512, 256), (256, 128), (128, 64)]]
        )
        self.up_sample = nn.ModuleList(
            [UpSample(i, o) for i, o in
              [(1024, 512), (512, 256), (256, 128), (128, 64)]]
        )
        self.concat = nn.ModuleList([Concat() for _ in range(4)])

        # Output
        self.last_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """ Forward pass of data x through UNet architecture.

        The tensor x is passes through each segment of the encoding and
        decoding portion of the network before a final 1x1 convolutional layer
        and sigmoidal activation returning a shape of [1 x 512 x 512]. A mask
        can be generated by thresholding the output at a desired value.

        Arguments
        ---------
            x (torch.Tensor): BATCH_SIZE x 1 x 512 x 512 grayscale image.

        Returns
        -------
            torch.Tensor: mask of the provided image.
        """
        through_pass = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            through_pass.append(x)
            x = self.down_sample[i](x)
        x = self.middle_conv(x)
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, through_pass.pop())
            x = self.up_conv[i](x)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        return x
    
    def train_model(
            self,
            train_loader: torch.utils.data.DataLoader,
            n_epochs: int,
            optimizer: torch.optim.Optimizer,
            loss_fn: Callable,
            verbose: bool=False,
            report_iter: int=25,
        ) -> None:
        """ Train the UNet model using the training data. 

        Arguments
        ---------
            train_loader (torch.utils.data.DataLoader): training data loader
            n_epochs (int): the number of training rounds through the dataset
            optimizer (torch.optim.Optimizer): optimizing function
            loss_fn (Callable): function to calculate loss
            verbose (bool): Indicates if function prints. Default is false.
            report_iter (int): Prints loss every nth batch. Default is 25.

        Returns
        -------
            None
        """
        self.train() # set the model to training mode
        for e in range(n_epochs):
            if verbose:
                print(f"EPOCH {e+1}")
            for i, batch in enumerate(train_loader):
                images = batch["image"]
                masks = batch["mask"]

                optimizer.zero_grad()
                y_pred = self(images)
                loss = loss_fn(y_pred, masks)
                loss.backward()
                optimizer.step()

                if verbose and i % report_iter == (report_iter - 1):
                    print(f" - batch {i+1} loss: {loss}")
        return None
        

class MiniUNet(nn.Module):
    """ A smaller UNet with fewer kernels. 
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Encoding arm
        self.down_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in
              [(in_channels, 16), (16, 32), (32, 64), (64, 128)]]
        )
        self.down_sample = nn.ModuleList(
            [DownSample() for _ in range(4)]
        )

        # Bottom of the U
        self.middle_conv = DoubleConvolution(128, 256)

        # Decoding arm
        self.up_conv = nn.ModuleList(
            [DoubleConvolution(i, o) for i, o in
              [(256, 128), (128, 64), (64, 32), (32, 16)]]
        )
        self.up_sample = nn.ModuleList(
            [UpSample(i, o) for i, o in
              [(256, 128), (128, 64), (64, 32), (32, 16)]]
        )
        self.concat = nn.ModuleList([Concat() for _ in range(4)])

        # Output
        self.last_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """ Forward pass of data x through UNet architecture.

        The tensor x is passes through each segment of the encoding and
        decoding portion of the network before a final 1x1 convolutional layer
        and sigmoidal activation returning a shape of [1 x 512 x 512]. A mask
        can be generated by thresholding the output at a desired value.

        Arguments
        ---------
            x (torch.Tensor): BATCH_SIZE x 1 x 512 x 512 grayscale image.

        Returns
        -------
            torch.Tensor: mask of the provided image.
        """
        through_pass = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            through_pass.append(x)
            x = self.down_sample[i](x)
        x = self.middle_conv(x)
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, through_pass.pop())
            x = self.up_conv[i](x)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        return x
    
    def train_model(
            self,
            train_loader: torch.utils.data.DataLoader,
            n_epochs: int,
            optimizer: torch.optim.Optimizer,
            loss_fn: Callable,
            verbose: bool=False,
            report_iter: int=25,
        ) -> None:
        """ Train the UNet model using the training data. 

        Arguments
        ---------
            train_loader (torch.utils.data.DataLoader): training data loader
            n_epochs (int): the number of training rounds through the dataset
            optimizer (torch.optim.Optimizer): optimizing function
            loss_fn (Callable): function to calculate loss
            verbose (bool): Indicates if function prints. Default is false.
            report_iter (int): Prints loss every nth batch. Default is 25.

        Returns
        -------
            None
        """
        self.train() # set the model to training mode
        for e in range(n_epochs):
            if verbose:
                print(f"EPOCH {e+1}")
            for i, batch in enumerate(train_loader):
                images = batch["image"]
                masks = batch["mask"]

                optimizer.zero_grad()
                y_pred = self(images)
                loss = loss_fn(y_pred, masks)
                loss.backward()
                optimizer.step()

                if verbose and i % report_iter == (report_iter - 1):
                    print(f" - batch {i+1} loss: {loss}")
        return None


class DoubleConvolution(nn.Module):
    """ Neural network layer that performs two convolutions with a 3x3 kernel,
        each followed by ReLU activation. 
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.a2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        """ Forward pass through a double convolution. """
        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        return x


class DownSample(nn.Module):
    """ Max pool with a kernel size and stride of 2. """
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """ Forward pass through 2D pooling layer. """
        return self.pool(x)


class Concat(nn.Module):
    """ Concatenates two tensors of similar shape along channel dimsension.

    Concat assumes the incoming data is of the shape [B x C x W x H] where B
    is the batches, C is the number of channels, and W and H are the width and
    height of the image. 
    """
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """ Forward pass through concatenation of two tensors. """
        return torch.concat([x, contracting_x], dim=1)


class UpSample(nn.Module):
    """ Performs upsampling of data with a kernel size and stride of 2. """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """ Forward pass through upsampling process. """
        return self.up(x)


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
