from collections import OrderedDict
import numpy as np

import scipy.sparse as sparse
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset


class DLSparseMaker(BaseEstimator):
    """
    This makes the race feature a one hot encoded value instread
    """

    def __init__(self, num_symptoms, age_mean=None, age_std=None):
        self.num_symptoms = num_symptoms
        self.age_std = age_std
        self.age_mean = age_mean

    def fit(self, df, y=None):
        self.age_mean = df['AGE'].mean()
        self.age_std = df['AGE'].std()

    def transform(self, df, y=None):
        if self.age_mean is None or self.age_std is None:
            raise ValueError("mean and std for age have not been evaluated. Have you run fit ?")
        symptoms = df.SYMPTOMS
        race = df.RACE

        df = df.drop(columns=['SYMPTOMS', 'RACE'])
        if 'LABEL' in df.columns:
            df = df[['LABEL', 'AGE', 'GENDER']]
        else:
            df = df[['AGE', 'GENDER']]

        df['AGE'] = (df['AGE'] - self.age_mean) / self.age_std

        dense_matrix = sparse.coo_matrix(df.values)
        symptoms = symptoms.apply(lambda v: [int(idx) + 5 for idx in v.split(",")])

        columns = []
        rows = []
        for idx, val in enumerate(symptoms):
            race_val = race.iloc[idx]
            rows += [idx] * (len(val) + 1)  # takes care of the race: it's one hot encoded, so!
            columns += [int(race_val)]
            columns += val

        data = np.ones(len(rows))
        symptoms_race_coo = sparse.coo_matrix((data, (rows, columns)), shape=(df.shape[0], self.num_symptoms + 5))
        data_coo = sparse.hstack([dense_matrix, symptoms_race_coo])
        return data_coo

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


class AiBasicMedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def data_to_tensor(self, data):
        return torch.FloatTensor(data.todense())

    def labels_to_tensor(self, labels):
        return torch.LongTensor(labels)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = self.labels_to_tensor(self.labels[idx])
        data = self.data_to_tensor(self.data[idx, :])

        return data, labels


class ClassificationBase(nn.Module):
    def step(self, batch):
        data, labels = batch
        out = self.forward(data)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        with torch.no_grad():
            acc = self.accuracy(out, labels)  # Calculate accuracy
        return loss, acc

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def summarize(self, losses, accuracies):
        loss = torch.stack(losses).mean()
        accuracy = torch.stack(accuracies).mean()

        return loss.item(), accuracy.item()

    def print(self, epoch, result):
        print("Epoch [{}],train_loss: {:.4f}, train_acc: {:.4f} val_loss: {:.4f}, val_acc: {:.4f}"
              .format(epoch, result['train_loss'], result['train_acc'], result['test_loss'], result['test_acc']))


class DNN(ClassificationBase):
    def __init__(self, input_dim, output_dim, layer_config=None, non_linearity='relu'):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_config = self.form_layer_config(layer_config)
        self.non_linearity = non_linearity
        self.model = self.compose_model()

    def get_default_config(self):
        return [
            [self.input_dim, 48],
            [48, 32],
            [32, 32],
            [32, self.output_dim]
        ]

    def form_layer_config(self, layer_config):
        if layer_config is None:
            return self.get_default_config()

        if len(layer_config) < 2:
            raise ValueError("Layer config must have at least two layers")

        if layer_config[0][0] != self.input_dim:
            raise ValueError("Input dimension of first layer config must be the same as input to the model")

        if layer_config[-1][1] != self.output_dim:
            raise ValueError("output dimension of last layer config must be the same as expected model output")

        for idx in range(len(layer_config) - 1):
            assert layer_config[idx][1] == layer_config[idx+1][0], "Dimension mismatch between layers %d and %d" % (idx, idx + 1)

        return layer_config

    def get_non_linear_class(self):
        if self.non_linearity == 'tanh':
            return nn.Tanh
        else:
            return nn.ReLU

    def compose_model(self):
        non_linear = self.get_non_linear_class()
        layers = OrderedDict()
        for idx in range(len(self.layer_config)):
            input_dim, output_dim = self.layer_config[idx]
            layers['linear-%d' % idx] = nn.Linear(input_dim, output_dim)
            if idx != len(self.layer_config) - 1:
                layers['nonlinear-%d' % idx] = non_linear()
        layers['final-soft-max'] = nn.Softmax(dim=1)
        return nn.Sequential(layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.model(x)
