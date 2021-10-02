#!/usr/bin/env python3


import cv2
import torch
import torch.nn.functional


# Initialize the board randomly
generation = 0
X_SIZE, Y_SIZE = (100, 100)
board = torch.rand(1, 1, X_SIZE, Y_SIZE)


# Initialize our convolution kernel
h = torch.tensor([[1.,  1.,  1.,],
                  [1.,  0.5, 1.,],
                  [1.,  1.,  1.,]]).view(1, 1, 3, 3)


while generation < 10000:
    generation += 1

    # Apply convolution
    board = torch.nn.functional.conv2d(board, h, padding=1)

    # Apply activation function
    board = torch.logical_and(board >= 2.5, board <= 3.5).float()

    # Draw the current iteration
    cv2.imshow('😎', cv2.resize(board.view(X_SIZE, Y_SIZE).numpy(), (400, 400)))
    cv2.waitKey(10)


cv2.waitKey(0)
