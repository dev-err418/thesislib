import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

import pathlib
import os
import pandas as pd
from timeit import default_timer as timer
import mlflow
import tempfile

from .utils import EarlyStopping, compute_accuracy, split_data, get_default_device, to_device, DeviceDataLoader, \
    compute_precision, compute_top_n, get_cnf_matrix
from .data import DLSparseMaker, AiBasicMedDataset
from .models import DNN, DEFAULT_LAYER_CONFIG


def dl_train(model, train_loader, optimizer):
    losses = []
    accs = []
    for batch, labels in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        with torch.no_grad():
            acc = compute_accuracy(out, labels)

        accs.append(acc)

    with torch.no_grad():
        batch_loss = torch.stack(losses).mean().item()
        batch_acc = torch.stack(accs).mean().item()
    return batch_loss, batch_acc


def dl_val(model, val_loader):
    losses = []
    accs = []
    with torch.no_grad():
        for batch, labels in val_loader:
            out = model(batch)
            loss = F.cross_entropy(out, labels)
            losses.append(loss)
            acc = compute_accuracy(out, labels)
            accs.append(acc)
        batch_loss = torch.stack(losses).mean().item()
        batch_acc = torch.stack(accs).mean().item()
    return batch_loss, batch_acc


def calculate_precision_accuracy_top_5(model, loader, num_labels, device=None):
    top_5_count = 0
    num_samples = 0
    cm = torch.zeros((num_labels, num_labels))
    unique_labels = torch.LongTensor(range(num_labels))
    if device is not None:
        cm = to_device(cm, device)
        unique_labels = to_device(unique_labels, device)

    with torch.no_grad():
        for item in loader:
            batch, y_true = item
            out = model(batch)
            y_pred = torch.max(out, dim=1)
            cnf = get_cnf_matrix(y_true, y_pred, unique_labels)
            cm.add_(cnf)
            top_5_acc, batch_samples = compute_top_n(out, y_true, 5)

            top_5_count += top_5_acc
            num_samples += batch_samples

        weighted_precision = compute_precision(cm)
        accuracy = torch.sum(torch.diagonal(cm))/num_samples
        top_5_accuracy = top_5_count/ num_samples

    return weighted_precision, accuracy.item(), top_5_accuracy


class MasterRunner:
    def __init__(self, train_file, **kwargs):
        self.train_file = train_file
        self.train_index_col = kwargs.get('index_col', "Index")
        self.train_split_size = kwargs.get('train_split_size', 0.8)
        self.num_symptoms = kwargs.get('num_symptoms', 376)
        self.num_conditions = kwargs.get('num_conditions', 801)
        self.input_dim = kwargs.get('input_dim', 383)
        self.train_batch_size = kwargs.get('train_batch_size', 128)
        self.val_batch_size = kwargs.get('val_batch_size', self.train_batch_size * 2)
        self.learning_rate = kwargs.get('learning_rate', 3e-3)
        self.device = get_default_device()
        self.layer_config = kwargs.get("layer_config", DEFAULT_LAYER_CONFIG)
        self.epochs = kwargs.get("epochs", 40)
        self.momentum = kwargs.get("momentum", 0.9)
        self.mlflow_uri = kwargs.get("mlflow_uri", None)
        self.mlflow_params = kwargs.get("mlflow_params", {})
        self.run_name = kwargs.get("run_name", "test_dl_run")

        self.early_stop = kwargs.get('early_stop', True)
        self.early_stop_patience = kwargs.get('early_stop_patience', 5)
        self.checkpoint_dir = kwargs.get('checkpoint_path', 'checkpoints')
        self.checkpoint_name = kwargs.get('checkpoint_dir', 'checkpoint.pt')
        self.has_scheduler = kwargs.get('has_scheduler', True)

        self.early_stopping = None
        if self.early_stop:
            pathlib.Path(self.checkpoint_dir).mkdir(exist_ok=True, parents=True)
            self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
            self.early_stopping = EarlyStopping(
                patience=self.early_stop_patience,
                verbose=False,
                path=self.checkpoint_path
            )

    def prep_data(self):
        df = pd.read_csv(self.train_file, index_col=self.train_index_col)
        labels = df.LABEL.values
        df = df.drop(columns=['LABEL'])

        train_data, train_labels, val_data, val_labels = split_data(df, labels, self.train_split_size)

        sparsifier = DLSparseMaker(self.num_symptoms)
        sparsifier.fit(train_data)

        train_data = sparsifier.transform(train_data)
        val_data = sparsifier.transform(val_data)

        train_data = AiBasicMedDataset(train_data, train_labels)
        val_data = AiBasicMedDataset(val_data, val_labels)

        train_loader = DataLoader(
            train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_data,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        train_loader = DeviceDataLoader(train_loader, self.device)
        val_loader = DeviceDataLoader(val_loader, self.device)

        return train_loader, val_loader

    def get_model(self):
        model = DNN(self.input_dim, self.num_conditions, self.layer_config)
        model = to_device(model, self.device)

        return model

    def run(self):
        metrics = {}

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.run_name)

        with mlflow.start_run():
            if len(self.mlflow_params) > 0:
                mlflow.log_params(self.mlflow_params)

            begin = timer()
            train_loader, val_loader = self.prep_data()
            metrics['prep_data'] = timer() - begin

            start = timer()
            model = self.get_model()
            metrics['compose_model'] = timer() - start

            train_accs = []
            train_losses = []
            val_accs = []
            val_losses = []

            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
            scheduler = None
            if self.has_scheduler:
                scheduler = StepLR(optimizer, step_size=15, gamma=0.1)

            start = timer()
            stopped_early = False
            for idx in range(self.epochs):
                train_loss, train_acc = dl_train(model, train_loader, optimizer)
                val_loss, val_acc = dl_val(model, val_loader)

                print("Epoch %d: Train: loss: %.5f\t acc: %.5f\nVal: loss: %.5f\t acc: %.5f" % (
                    idx + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc
                ))

                train_accs.append(train_acc)
                train_losses.append(train_loss)
                val_accs.append(val_acc)
                val_losses.append(val_loss)

                if self.early_stop:
                    self.early_stopping(val_loss, model, idx)

                    if self.early_stopping.early_stop:
                        stopped_early =True
                        break

                if scheduler is not None:
                    scheduler.step()

            metrics['train_time'] = timer() - start

            if stopped_early:
                state_dict = torch.load(self.early_stopping.path)
                model.load_state_dict(state_dict)
            else:
                state_dict = model.state_dict()

            # need to calculate the precision, and top_5 accuracy on train and val set
            start = timer()
            train_precision, train_accuracy, train_top_5 = calculate_precision_accuracy_top_5(
                model,
                train_loader,
                self.num_conditions,
                self.device
            )
            metrics['train_score_time'] = timer() - start

            start = timer()
            val_precision, val_accuracy, val_top_5 = calculate_precision_accuracy_top_5(
                model,
                val_loader,
                self.num_conditions,
                self.device
            )
            metrics['train_score_time'] = timer() - start

            metrics['train_precision'] = train_precision
            metrics['train_accuracy'] = train_accuracy
            metrics['val_precision'] = val_precision
            metrics['val_accuracy'] = val_accuracy
            metrics['run_time'] = timer() - begin

            # save results ! mlflow?
            artifacts = {
                'train_loss': train_losses,
                'train_accuracies': train_accs,
                'val_loss': val_losses,
                'val_accuracies': val_accs,
                'model_dict': state_dict,
                'num_conditions': self.num_conditions,
                'num_symptoms': self.num_symptoms,
                'layer_config': self.layer_config
            }

            # log metrics
            mlflow.log_metrics(metrics)

            # log artifacts
            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = os.path.join(tmpdirname, "dl_model.torch")
                torch.save(artifacts, filename)
                mlflow.log_artifact(filename, 'dl_model.torch')


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
