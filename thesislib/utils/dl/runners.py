import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pathlib
import os

from .utils import EarlyStopping


class Runner:
    def __init__(self, model, train_loader, test_loader, **kwargs):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimiser_name = kwargs.get('optimiser_name', "sgd")
        self.optimiser_params = kwargs.get('optimiser_params', {})
        self.lr_start = kwargs.get('lr_start', 0.0001)
        self.lr = self.lr_start
        self.visdom = kwargs.get('visdom', None)
        self.epochs = kwargs.get('epochs', 200)
        self.early_stop = kwargs.get('early_stop', True)
        self.checkpoint_dir = kwargs.get('checkpoint_path', 'checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_dir', 'checkpoint.pt')
        self.has_scheduler = kwargs.get('has_scheduler', True)
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
                patience=10,
                threshold=1e-5
            )
        else:
            self.scheduler = None

        self.early_stopping = None
        if self.early_stop:
            pathlib.Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)
            self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
            self.early_stopping = EarlyStopping(patience=20, verbose=False, path=self.checkpoint_path)

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.final_epoch = None

    def get_optimiser(self):
        if self.optimiser_name == 'sgd':
            optimiser = optim.RMSprop
        elif self.optimiser_name == 'adam':
            optimiser = optim.Adam
        else:
            optimiser = optim.RMSprop

        return optimiser

    def run(self, is_train=True):
        loader = self.train_loader if is_train else self.test_loader

        losses = []
        accuracies = []

        for batch in loader:
            if is_train:
                self.optimiser.zero_grad()
                loss, acc = self.model.step(batch)
                self.run_counter += 1
                loss.backward()
                self.optimiser.step()
            else:
                with torch.no_grad():
                    loss, acc = self.model.step(batch)

            losses.append(loss)
            accuracies.append(acc)

        return losses, accuracies

    def plot_visdom(self, results):
        if self.visdom is None:
            return None

        epoch = results.get('epoch')
        train_loss = results.get('train_loss')
        val_loss = results.get('val_loss')
        train_acc = results.get('train_acc')
        val_acc = results.get('val_acc')

        update = None if epoch == 0 else 'append'

        self.visdom.line([train_loss], [epoch], update=update, opts={
            'title': 'Train Loss',
            'xlabel': 'Epochs',
            'ylabel': 'Loss (cross entropy)',
        }, win='ai_med_train_loss')

        self.visdom.line([train_acc], [epoch], update=update, opts={
            'title': 'Train Accuracy',
            'xlabel': 'Epochs',
            'ylabel': 'Accuracy',
        }, win='ai_med_train_acc')

        self.visdom.line([val_loss], [epoch], update=update, opts={
            'title': 'Validation Loss',
            'xlabel': 'Epochs',
            'ylabel': 'Loss (cross entropy)',
        }, win='ai_med_test_loss')

        self.visdom.line([val_acc], [epoch], update=update, opts={
            'title': 'Validation Accuracy',
            'xlabel': 'Epochs',
            'ylabel': 'Accuracy',
        }, win='ai_med_test_acc')

        return True

    def fit(self):
        self.final_epoch = None
        for epoch in range(self.epochs):
            train_losses, train_accuracies = self.run()
            val_losses, val_accuracies = self.run(is_train=False)

            train_loss, train_accuracy = self.model.summarize(train_losses, train_accuracies)
            val_loss, val_accuracy = self.model.summarize(val_losses, val_accuracies)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            result = {
                'train_loss': train_loss,
                'train_acc': train_accuracy,
                'val_loss': val_loss,
                'val_acc': val_accuracy,
                'epoch': epoch
            }

            self.train_loss.append(train_loss)
            self.train_acc.append(train_accuracy)
            self.val_loss.append(val_loss)
            self.val_acc.append(val_accuracy)

            self.model.print(epoch, result)

            self.plot_visdom(result)
            will_break = False

            if self.early_stop:
                self.early_stopping(val_loss, self.model, epoch+1)

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
