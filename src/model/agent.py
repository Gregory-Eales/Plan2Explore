import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl

from .policy_net import PolicyNetwork
from .value_net import ValueNetwork


class Agent(pl.LightningModule):

    def __init__(self, hparams=None, in_dim=None, out_dim=None):
        super(Agent, self).__init__()

        self.hparams = hparams

        self.policy_net = PolicyNetwork(in_dim=in_dim, out_dim=out_dim)
        self.value_net = ValueNetwork(in_dim=in_dim, out_dim=out_dim)
        self.rssm = RSSM(in_dim=in_dim, out_dim=out_dim)

        self.image_encoder = None
        self.image_decoder = None

    def act(self, s):
        pass

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        
        # STAGE 1: Dataset from Random Policy
        if self.random_collection:
            pass

        # STAGE 2: EXPLORE
        if self.exploring:

            # TRAIN WORLD MODEL (M) ON DATASET (D)

            # TRAIN LATENT DISAGREEMENT ENSAMBLE (E) ON DATASET (D)

            # TRAIN POLICY ON LATENT DISAGREEMENT REWARD IN IMAGINATION of M

            # EXECUTE POLICY IN ENVIRONMENT TO EXPAND DATASET (D)


            x, y = batch
            y_hat = self.forward(x)
            loss = F.cross_entropy(y_hat, y)

            tensorboard_logs = {'train_loss': loss}

            return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_val_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser

