#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
### YOUR CODE HERE for part 1h


class Highway(nn.Module):
    """it is not fun"""
    def __init__(self, conv_size, hidden_size, high_size):
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(conv_size, hidden_size)
        self.w_gate = nn.Linear(hidden_size, high_size)

    def forward(self, x_conv_out):
        """
        Map from x_conv_out to x_highway with batches
        @para a: object
        @return x_highway
        """
        x_highway = self.linear1(x_conv_out)
        x_highway = self.linear2(x_highway)
        return x_highway

### END YOUR CODE 

