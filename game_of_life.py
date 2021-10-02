#!/usr/bin/env python3

from absl import app
from absl import flags

import csv
import cv2
import torch
import torch.nn.functional

FLAGS = flags.FLAGS

flags.DEFINE_integer('generations', 10000, 'Number of iterations')
flags.DEFINE_string(
    'board_file', None, 'If specified, read the initial board state from this '
    'file, instead of generating a random one.')

def main(argv):
    # Initialize the board randomly
    generation = 0
    X_SIZE, Y_SIZE = (100, 100)
    if FLAGS.board_file:
        from_file = []
        X_SIZE = 0
        Y_SIZE = 0
        with open(FLAGS.board_file) as file:
            csv_reader =  csv.reader(file)
            for row in csv_reader:
                from_file.append(row)
                X_SIZE += 1
                Y_SIZE = max(len(row), Y_SIZE)
        board = torch.empty(1, 1, X_SIZE, Y_SIZE)
        for i in range(X_SIZE):
            for j in range(Y_SIZE): 
                board[0, 0, i, j] = int(from_file[i][j]) if len(from_file[i]) > j else 0
                    
    else:
        board = torch.rand(1, 1, X_SIZE, Y_SIZE)
    
    
    # Initialize our convolution kernel
    h = torch.tensor([[1.,  1.,  1.,],
                      [1.,  0.5, 1.,],
                      [1.,  1.,  1.,]]).view(1, 1, 3, 3)
    
    
    while generation < FLAGS.generations:
        # Apply convolution
        board = torch.nn.functional.conv2d(board, h, padding=1)
    
        # Apply activation function
        board = torch.logical_and(board >= 2.5, board <= 3.5).float()
    
        # Draw the current iteration
        cv2.imshow('😎', cv2.resize(board.view(X_SIZE, Y_SIZE).numpy(), (400, 400)))
        cv2.waitKey(10)
    
    
    cv2.waitKey(0)

app.run(main)
