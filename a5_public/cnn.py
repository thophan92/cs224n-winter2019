#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    """
    To map from x_reshaped to x_conv_out
    """
    def __init__(self, emb_char_size, emb_word_size, max_word_length, kernel_size=5, stride=1):
        """
        :param emb_char_size: in_channels
        :param emb_word_size: out_channels (number of channels/filters)
        :param kernel_size: default 5
        :param stride: default 1
        """
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=emb_char_size, out_channels=emb_word_size,
                                kernel_size=kernel_size, stride=stride, bias=True)
        self.relu = F.relu
        self.max_pool = nn.MaxPool1d(max_word_length - kernel_size + 1, stride=1)

    def forward(self, x_reshaped):
        """
        from x_resahped to x_conv_out
        :param x_reshaped: shape - (batch_size, emb_char_size, max_word_length)
        :return x_conv_out: shape - (batch_size, emb_word_size)
        """
        x_conv = self.conv1d(x_reshaped) # shape (batch_size, emb_word_size, max_word_length - 4)
        x_conv_out = self.max_pool(self.relu(x_conv)).squeeze() # shape (batch_size, emb_word_size, 1) --> (batch_size, emb_word_size)
        return x_conv_out

### END YOUR CODE


if __name__ == '__main__':
    emb_char_size, emb_word_size, max_word_length = 5, 10, 8
    cnn = CNN(emb_char_size, emb_word_size, max_word_length)
    x_reshaped = torch.randn(10, emb_char_size, max_word_length)
    x_conv_out = cnn(x_reshaped)

    assert x_conv_out.shape == (10, emb_word_size)
    print("Successful")