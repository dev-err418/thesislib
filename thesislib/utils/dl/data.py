from collections import OrderedDict
import numpy as np

import scipy.sparse as sparse
from sklearn.base import BaseEstimator

import torch

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
        return sparse.hstack([dense_matrix, symptoms_race_coo]).tocsc()

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


class AiBasicMedDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = self.data_to_tensor(data)
        self.labels = self.labels_to_tensor(labels)
        self.transform = transform

    def data_to_tensor(self, data):
        return torch.FloatTensor(data.todense())

    def labels_to_tensor(self, labels=None):
        if labels is None:
            return None

        return torch.LongTensor(labels)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx, :]
        if self.labels is not None:
            labels = self.labels[idx]
            return data, labels

        return data


class AiDAEMedDataset(AiBasicMedDataset):
    def __init__(self, data, labels, dae):
        super(AiDAEMedDataset, self).__init__(data, labels=labels)
        self.dae = dae

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx, :]
        labels = self.labels[idx]

        with torch.no_grad():
            if len(data.shape) == 1:
                data = data.view(1, -1)
                compressed = self.dae.encoder(data[:, 2:])
                combined = torch.cat([data[:, :2], compressed], dim=1)
                data = combined.view(-1)
            else:
                compressed = self.dae.encoder(data[:, 2:])
                data = torch.cat([data[:, :2], compressed], dim=1)

        return data, labels

