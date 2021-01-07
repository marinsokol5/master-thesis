import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from constants import paths as p
from constants import tokens as t
from constants import hyperparameters as hp


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout_probability=0.1, cached_maximum_sequence_length=5_000):
        super().__init__()
        self.model_dimension = model_dimension
        self.cached_maximum_sequence_length = cached_maximum_sequence_length

        positional_encoding = self._compute_positional_encoding_matrix(model_dimension, cached_maximum_sequence_length)
        # positional_encoding -> cached_maximum_sequence_length x model_dimension
        self.register_buffer('positional_encoding', positional_encoding)

        self.dropout_layer = nn.Dropout(dropout_probability)

    def _compute_positional_encoding_matrix(self, model_dimension, sequence_length):
        first_multiplier = torch.arange(0, sequence_length, dtype=torch.float)
        second_multiplier = 1.0 / torch.pow(torch.tensor(10_000),
                                            torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)
        computation = rearrange(first_multiplier, 'f -> f 1') * rearrange(second_multiplier, 's -> 1 s')

        positional_encoding = torch.zeros(sequence_length, model_dimension)
        positional_encoding[:, ::2] = torch.sin(computation)
        positional_encoding[:, 1::2] = torch.cos(computation)

        return positional_encoding

    def forward(self, batched_input):
        # batched_input -> batch_size x sequence_length x model_dimension
        positional_encoding = self.positional_encoding if (batched_input.shape[
                                                               -2] <= self.cached_maximum_sequence_length) else self._compute_positional_encoding_matrix(
            self.model_dimension, batched_input.shape[-2])

        positionaly_encoded_input = batched_input + positional_encoding[:batched_input.shape[-2], :]
        return self.dropout_layer(positionaly_encoded_input)
