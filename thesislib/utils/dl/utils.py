import torch
import torch.nn.functional as F

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Source: https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.epoch = None

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class VisdomConfig:
    url = None
    port = None
    username = None
    password = None
    env = None


def split_data(data, labels, train_size):
    split_selector = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size
    )

    train_data = None
    val_data = None
    train_labels = None
    val_labels = None
    for train_index, val_index in split_selector.split(data, labels):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        train_labels = labels[train_index]
        val_labels = labels[val_index]

    return train_data, train_labels, val_data, val_labels


def get_cnf_matrix(y_true, y_pred, labels):
    """
    Given the target and actual predictions estimate a confusion matrix.
    In addition to the confusion matrix being useful on its own, this is a good starting point
    for computing parameters like precision, recall, support, etc

    In the case of pytorch where the results (most likely) come in batches, the confusion matrix for the whole dataset
    is obtained by summing the confusion matrix from each batch
    :param y_true: torch.LongTensor(shape=(n_samples,))
    :param y_pred: torch.LongTensor(shape=(n_samples,))
    :return: torch.LongTensor(shape=(n_labels, n_labels)
    """
    sample_weight = torch.ones(y_true.shape[0], dtype=torch.int64)

    label_to_ind = {y.item(): x for x, y in enumerate(labels)}

    n_labels = labels.shape.numel()
    _y_pred = y_pred.new_tensor([label_to_ind.get(x.item(), n_labels + 1) for x in y_pred])
    _y_true = y_true.new_tensor([label_to_ind.get(x.item(), n_labels + 1) for x in y_true])

    ind = torch.logical_and(_y_pred < n_labels, y_true < n_labels)

    _y_pred = _y_pred[ind]
    _y_true = _y_true[ind]
    sample_weight = sample_weight[ind]

    cm_ind = torch.cat((_y_true.view(1, -1), _y_pred.view(1, -1)), dim=0)

    # this works because when constructing sparse matrices, duplicate entries are summed up
    # just in case you're scratching your head at this again!
    cm = torch.sparse.LongTensor(cm_ind, sample_weight, torch.Size([n_labels, n_labels]))

    return cm.to_dense()


def compute_accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def compute_precision(confusion_matrix):
    """
    Computes the weighted precision given a confusion matrix

    A pytorch minimalistic implementation of the sklearn.metrics.confusion_matrix function

    No checks are made for zero division and validity of the confusion matrix

    :param confusion_matrix: torch.Tensor(shape=(n_labels, n_labels))
    :return: float
    """

    true_positives = torch.diagonal(confusion_matrix)

    false_positives = torch.sum(confusion_matrix, dim=0) - true_positives

    unweighted_precision = torch.true_divide(true_positives, true_positives + false_positives)

    # to get the weighted precision, we use the label count as weights
    label_counts = torch.sum(confusion_matrix, dim=1)
    num_samples = torch.sum(label_counts)

    weighted_precision = torch.true_divide(unweighted_precision * label_counts, num_samples)

    return weighted_precision.item()


def compute_top_n(out, labels, n):
    """
    Computes the top_n accuracy.
    Note that the labels must have been properly encoded. So if your set of real labels are {8, 100, 7}
    then you'd want to pass {0, 1, 2} where {0, 1, 2} maps to {8, 100, 7}
    :param out: torch.FloatTensor(shape=(num_samples, num_labels)). This is the probablity estimate
    (or some monotonically increasing version of it)
    :param labels: torch.FloatTensor(shape=(num_samples)) the target labels
    :param n: int.
    :return: (num_top_n_accurate, num_samples)
    """
    sorted_prob = torch.argsort(out, dim=1, descending=True)
    top_n = sorted_prob[:, :n]

    combined = top_n == labels.view(1, -1)
    top_n_accurate = torch.sum(combined).item()
    num_samples = labels.shape[0]

    return top_n_accurate, num_samples
