#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.emb_char_size = 50
        self.embed_size = embed_size
        self.max_word_len = 21
        pad_token_idx = vocab.char2id['<pad>']
        # self.dropout_rate = 0.5

        self.char_embeddings = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim=self.emb_char_size,
                                            padding_idx=pad_token_idx)
        self.cnn = CNN(emb_char_size=self.emb_char_size, emb_word_size=self.embed_size,
                       max_word_length=self.max_word_len)
        self.highway = Highway(e_word_size=self.embed_size)
        # self.dropout = nn.Dropout(self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        embed = self.char_embeddings(input) # shape: (sentence_length, batch_size, max_word_length, emb_char_size)
        (sentence_length, batch_size, max_word_length, _) = embed.shape
        embed = embed.view((sentence_length * batch_size, max_word_length, -1)) # (sentence_length * batch_size, max_word_length, emb_char_size)
        embed = embed.transpose(1, 2) # (b, emb_char_size, max_word_length)
        x_conv_out = self.cnn(embed) # (b, emb_word_size)
        x_highway = self.highway(x_conv_out) # (b, emb_word_size)
        output = x_highway.view(sentence_length, batch_size, -1) # (sentence_length, batch_size, emb_word_size)
        return output
        ### END YOUR CODE
