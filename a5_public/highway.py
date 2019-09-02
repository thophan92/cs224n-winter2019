#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
### YOUR CODE HERE for part 1h
import torch

class Highway(nn.Module):
    """it is not fun"""
    def __init__(self, e_word_size, drop_rate=0.3):
        super(Highway, self).__init__()
        self.w_proj = nn.Linear(e_word_size, e_word_size)
        self.w_gate = nn.Linear(e_word_size, e_word_size)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.sigmoid # nn.functional.sigmoid is deprecated
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x_conv_out):
        """
        Map from x_conv_out to x_highway with batches
        @para x_conv_out: shape (b, e_word_size): b - batch size, e_word_size
        @return x_highway: shape (b, e_word_size)
        """
        x_proj = self.relu(self.w_proj(x_conv_out)) # (b, 50)
        x_gate = self.sigmoid(self.w_gate(x_proj)) # (b, e_word_size)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return self.dropout(x_highway)

### END YOUR CODE 

if __name__ == '__main__':
    emb_word_size = 21
    highway = Highway(emb_word_size)
    batch_size = 10
    x_conv_out = torch.randn(batch_size, emb_word_size)
    x_highway = highway(x_conv_out)
    assert x_highway.shape == (batch_size, emb_word_size)
    print("Successful")
