import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.positional_encoding import PositionalEncoding

from constants import paths as p
from constants import tokens as t
from constants import hyperparameters as hp


class Transformer(nn.Module):
    def __init__(self, vocabulary_size=hp.VOCABULARY_SIZE, embedding_size=hp.EMBEDDING_SIZE,
                 number_of_properties=hp.NUMBER_OF_PROPERTIES, padding_index=hp.PADDING_INDEX,
                 model_dimension=hp.MODEL_DIMENSION, target_sequence_length=hp.TARGET_SEQUENCE_LENGTH,
                 dropout_probability=hp.DROPOUT_PROBABILITY,
                 feed_forward_transformer_layer_dimension=hp.FEED_FORWARD_TRANSFORMER_LAYER_DIMENSION):
        super().__init__()

        self.model_dimension = model_dimension

        self.embedder = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_index)
        self.embedder_drouput = nn.Dropout(dropout_probability)
        self.embedding_layer = nn.Sequential(self.embedder, self.embedder_drouput)

        self.combine_embeddings_and_properties_layer = nn.Linear(embedding_size + number_of_properties, model_dimension)
        self.positional_encoding_layer = PositionalEncoding(model_dimension, dropout_probability=dropout_probability)
        self.transformer = nn.Transformer(
            d_model=model_dimension,
            dim_feedforward=feed_forward_transformer_layer_dimension,
            dropout=dropout_probability
        )
        self.transformer_to_vocabulary_logits_layer = nn.Linear(model_dimension, vocabulary_size)

        self.register_buffer(
            'decoder_attention_mask',
            torch.full((target_sequence_length, target_sequence_length), float("-inf")).triu(diagonal=1)
        )

    def forward(self, batched_source_numericalized, batched_source_properties, batched_target_input_numericalized,
                batched_source_padding_mask, batched_target_input_padding_mask):
        # all needs to be torch.float, torch.double is not supported by Transformer()
        # batched_source_numericalized -> batch_size x source_sequence_length(start_id, t1, t2, t3.., end_id, start_id, t4.., pad_id, pad_id..)(int/long)
        # batched_source_properties -> batch_size x source_sequence_length x 5(number_of_properties)(0 for special tokens...)(float)
        # batched_target_input_numericalized -> batch_size x target_sequence_length (start_id, t1, t2... pad_id, pad_id, pad_id...)(int/long)(without end_id)
        # batched_source/target_padding_mask -> batch_size x source/target_sequence_length(False, False..., True, True..)(bool)

        batched_source_embedded = self.embedding_layer(batched_source_numericalized)
        batched_target_embedded = self.embedding_layer(batched_target_input_numericalized)
        # batched_source/target_embedded -> batch_size x source/target_sequence_length x embedding_size

        batched_source = torch.cat((batched_source_embedded, batched_source_properties),
                                   dim=2)  # batch_size x source_sequence_length x embedding_size + number_of_properies
        batched_source = self.combine_embeddings_and_properties_layer(
            batched_source)  # batch_size x source_sequence_length x model_dimension

        batched_source = self.positional_encoding_layer(batched_source * math.sqrt(
            self.model_dimension))  # normalizing(reducing variance) before positonally encoding
        batched_target = self.positional_encoding_layer(batched_target_embedded * math.sqrt(self.model_dimension))
        # batched_source/target -> batch_size x source/target_sequence_length x model_dimension

        transformer_output = self.transformer(
            src=rearrange(batched_source, 'b s m -> s b m'),  # batched_source.tranpose(0, 1)
            tgt=rearrange(batched_target, 'b s m -> s b m'),
            tgt_mask=self.decoder_attention_mask,
            src_key_padding_mask=batched_source_padding_mask,
            memory_key_padding_mask=batched_source_padding_mask,
            tgt_key_padding_mask=batched_target_input_padding_mask,
        )
        transformer_output = rearrange(transformer_output, 's b m -> b s m')
        # transformer_output -> batch_size x target_sequence_length x model_dimension

        vocabulary_logits = self.transformer_to_vocabulary_logits_layer(transformer_output)
        # vocabulary_logits -> batch_size x target_sequence_length x vocabulary_size

        # vocabulary_log_probability = F.log_softmax(vocabulary_logits, dim=-1)
        # it's more numerically stable
        # https://deepdatascience.wordpress.com/2020/02/27/log-softmax-vs-softmax/

        return vocabulary_logits
