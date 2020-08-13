"""
A deep auto-encoder for attempting dimension reduction on the symptoms.

The aim is to try and get some latent continuous representation of the symptoms
And use this as some pre-training step.

We're looking for a compression. So we shrink when we encode and expand when we decode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import EarlyStopping

import pathlib
import os


class DAE (nn.Module):
    def __init__(self, **kwargs):
        super(DAE, self).__init__()

        self.input_dim = kwargs.get("input_dim", 376)
        self.target_dim = kwargs.get("target_dim", 64)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=self.target_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.target_dim, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=256),
            nn.Tanh(),
            nn.Linear(in_features=256, out_features=self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.decoder(self.encoder(data))


class DAERunner:
    def __init__(self, model, train_loader, test_loader, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimiser_name = kwargs.get('optimiser_name', "sgd")
        self.optimiser_params = kwargs.get('optimiser_params', {})
        self.lr = kwargs.get('lr', 0.001)
        self.visdom = kwargs.get('visdom', None)
        self.epochs = kwargs.get('epochs', 200)
        self.early_stop = kwargs.get('early_stop', True)
        self.early_stop_patience = kwargs.get('early_stop_patience', 20)
        self.checkpoint_dir = kwargs.get('checkpoint_path', 'checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_dir', 'checkpoint.pt')
        self.has_scheduler = kwargs.get('has_scheduler', True)
        self.scheduler_patience = kwargs.get('scheduler_patience', 10)
        self.scheduler_threshold = kwargs.get('scheduler_threshold', 1e-4)
        self.checkpoint_path = None
        self.run_counter = 0
        self.train_loss = []
        self.validation_loss = []
        self.optimiser_cls = self.get_optimiser()
        self.optimiser = self.optimiser_cls(self.model.parameters(), lr=self.lr, **self.optimiser_params)

        if self.has_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.optimiser,
                mode='min',
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold
            )
        else:
            self.scheduler = None

        self.early_stopping = None
        if self.early_stop:
            pathlib.Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)
            self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
            self.early_stopping = EarlyStopping(patience=self.early_stop_patience, verbose=False, path=self.checkpoint_path)

        self.train_loss = []
        self.val_loss = []
        self.final_epoch = None

    def get_optimiser(self):
        if self.optimiser_name == 'sgd':
            optimiser = optim.RMSprop
        elif self.optimiser_name == 'adam':
            optimiser = optim.Adam
        else:
            optimiser = optim.RMSprop

        return optimiser

    def step(self, batch):
        data = batch
        out = self.model(data)  # Generate reconstructed output
        loss = F.binary_cross_entropy(out, data)
        return loss

    def run(self, is_train=True):
        loader = self.train_loader if is_train else self.test_loader

        losses = []

        for batch in loader:
            if is_train:
                loss = self.step(batch)
                self.run_counter += 1
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
            else:
                with torch.no_grad():
                    loss = self.step(batch)

            losses.append(loss.detach())

        return losses

    def plot_visdom(self, results):
        if self.visdom is None:
            return None

        epoch = results.get('epoch')
        train_loss = results.get('train_loss')
        val_loss = results.get('val_loss')

        update = None if epoch == 0 else 'append'

        self.visdom.line([train_loss], [epoch], update=update, opts={
            'title': 'Train Loss',
            'xlabel': 'Epochs',
            'ylabel': 'Loss (Binary cross entropy)',
        }, win='ai_med_train_loss')

        self.visdom.line([val_loss], [epoch], update=update, opts={
            'title': 'Validation Loss',
            'xlabel': 'Epochs',
            'ylabel': 'Loss (cross entropy)',
        }, win='ai_med_test_loss')

        return True

    def print(self, epoch, result):
        print("Epoch [{}],train_loss: {:.4f}, val_loss: {:.4f}"
              .format(epoch, result['train_loss'], result['val_loss']))

    def fit(self):
        self.final_epoch = None
        for epoch in range(self.epochs):
            train_losses = self.run()
            val_losses = self.run(is_train=False)

            train_losses = torch.stack(train_losses).mean()
            val_losses = torch.stack(val_losses).mean()

            if self.scheduler is not None:
                self.scheduler.step(val_losses)

            result = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'epoch': epoch
            }

            self.train_loss.append(train_losses)
            self.val_loss.append(val_losses)

            self.print(epoch, result)

            self.plot_visdom(result)
            will_break = False

            if self.early_stop:
                self.early_stopping(val_losses, self.model, epoch+1)

                if self.early_stopping.early_stop:
                    will_break = True

            if will_break:
                print("Early Stopping at : Epoch: %d" % epoch)
                self.final_epoch = epoch + 1
                break
        # set the final epoch
        if self.final_epoch is None:
            self.final_epoch = self.epochs # we got to the end

    def get_epoch(self):
        return self.final_epoch
