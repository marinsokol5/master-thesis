import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from einops import rearrange

from constants import paths as p
from constants import tokens as t
from constants import hyperparameters as hp

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
from models.transformer import Transformer



class IMDB_Reviews(pl.LightningDataModule):
    def __init__(self, train_path=p.TRAIN_TENSOR_DATASET_PATH, validaiton_path=p.VALIDATION_TENSOR_DATASET_PATH, batch_size=hp.BATCH_SIZE):
        super().__init__()
        self.train_path = train_path
        self.validation_path = validaiton_path
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_data = torch.load(self.train_path)
        self.validation_data = torch.load(self.validation_path)

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, self.batch_size)


class TransformerLightning(pl.LightningModule):
    def __init__(self, vocabulary_size=hp.VOCABULARY_SIZE, embedding_size=hp.EMBEDDING_SIZE,
                 number_of_properties=hp.NUMBER_OF_PROPERTIES, padding_index=hp.PADDING_INDEX,
                 model_dimension=hp.MODEL_DIMENSION, target_sequence_length=hp.TARGET_SEQUENCE_LENGTH,
                 dropout_probability=hp.DROPOUT_PROBABILITY,
                 feed_forward_transformer_layer_dimension=hp.FEED_FORWARD_TRANSFORMER_LAYER_DIMENSION):
        super().__init__()
        self.transformer = Transformer(vocabulary_size, embedding_size, number_of_properties, padding_index,
                                       model_dimension, target_sequence_length, dropout_probability,
                                       feed_forward_transformer_layer_dimension)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=padding_index)

        self.train_name = "train"
        self.validation_name = "validation"

    def forward(self, batched_source_numericalized, batched_source_properties, batched_source_padding_mask):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=6 * 10e-5)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, self.train_name)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, self.validation_name)

    def _step(self, batch, batch_idx, mode):
        bsn, bsp, btin, bspm, btipm, bton = batch
        logits = self.transformer(bsn, bsp, btin, bspm, btipm)

        # logits -> batch_size x target_sequence_length x vocabulary_size
        # bton -> batch_size x target_sequence_length([0, vocabulary_size-1])
        loss = self.loss_function(rearrange(logits, 'b t v -> b v t'), bton)

        on_step = True if mode == self.train_name else False
        self.log(f'{mode}_loss', loss, on_step=on_step, on_epoch=True, prog_bar=True, logger=True)

        return loss

