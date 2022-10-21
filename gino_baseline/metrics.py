import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.signal as signal
from skimage.metrics import hausdorff_distance
import SimpleITK as sitk


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:

    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss."""
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)



def dice_loss(
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return 1-DiceLoss()(input, target)

def generate_circle(size):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    Z = Z < 0.5
    Z = Z.astype(np.float32)
    return Z

def generate_square(size):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    Z = np.ones((size, size))
    Z[(X < -0.5) | (X > 0.5)] = 0
    Z[(Y < -0.5) | (Y > 0.5)] = 0
    Z = Z.astype(np.float32)
    return Z

def edge_filter_mask(image):
    # Define the Sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # Apply the filter on the image
    image_x = signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    image_y = signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    # Compute the magnitude
    magnitude = np.sqrt(np.square(image_x) + np.square(image_y))

    # Compute the outline
    outline = np.zeros_like(magnitude)
    outline[magnitude > 0] = 1

    return outline

def convert_to_bool(image):
    image = image.astype(bool)
    return image

def dice_score(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def ejection_fraction(ES, ED):
    projection_area_ES = np.sum(ES)
    projection_area_ED = np.sum(ED)
    return 1-(projection_area_ES/projection_area_ED)**(3/2)

def calculate_hausdorff_distance(y_hat, y):
    distances = [[],[],[],[]]
    y = torch.nn.functional.one_hot(y, num_classes=4)
    y = torch.permute(y, (0,3,1,2)).numpy()
    # argmax to get the class with the highest probability and one hot encode it
    y_hat = torch.nn.functional.one_hot(torch.argmax(y_hat, dim=1), num_classes=4)
    y_hat = torch.permute(y_hat, (0,3,1,2)).numpy()

    for area in range(y.shape[1]):
        for mask in range(y.shape[0]):
            y_temp = y[mask][area]
            y_hat_temp = y_hat[mask][area]

            outline_y = edge_filter_mask(y_temp)
            y_temp = convert_to_bool(outline_y)

            outline_y_hat = edge_filter_mask(y_hat_temp)
            y_hat_temp = convert_to_bool(outline_y_hat)

            distance = hausdorff_distance(y_hat_temp, y_temp)
            if distance < 1e10:
                distances[area].append(distance)

    return [np.mean(distances[0]), np.mean(distances[1]), np.mean(distances[2]), np.mean(distances[3])]