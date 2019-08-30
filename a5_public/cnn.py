#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn


class CNN(nn.Module):
    """
    To map from x_reshaped to x_conv_out
    """
    def __init__(self, kernel_size=5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(kernel_size=kernel_size)

    def forward(self, x_reshaped):
        pass

### END YOUR CODE

