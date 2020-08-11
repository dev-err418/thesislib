from thesislib.utils.dl.utils import VisdomConfig, get_default_device, DeviceDataLoader, to_device
from thesislib.utils.dl.models import DLSparseMaker, AiBasicMedDataset, DNN
from thesislib.utils.dl.runners import Runner

from torch.utils.data import DataLoader
import torch
import json

import os
import botocore.session
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import visdom
import time
import math
import mlflow
from timeit import default_timer as timer

S3_BUCKET = "qcedelft"

AWS_REGION = "us-east-1"


class Bench:
    def __init__(self, run_name, train_file, mlflow_uri, input_dim, num_symptoms, num_conditions, **kwargs):
        self.train_file = train_file
        self.num_symptoms = num_symptoms
        self.num_conditions = num_conditions
        self.input_dim = input_dim
        self.run_name = run_name

        self.train_size = kwargs.get("train_size", 0.8)
        self.epochs = kwargs.get("epochs", 200)
        self.random_state = kwargs.get("random_state", None)
        self.visdom_config = kwargs.get("visdom_config", None)
        self.train_batch_size = kwargs.get("train_batch_size", 128)
        self.val_batch_size = kwargs.get("val_batch_size", 128)
        self.layer_config_file = kwargs.get("layer_config_file", None)
        self.non_linearity = kwargs.get("non_linearity", 'relu')
        self.tmp_directory = kwargs.get("tmp_dir", "/tmp")
        self.mlflow_uri = mlflow_uri

        self.age_std = None
        self.age_mean = None
        self.runner = None
        self.run_metrics = {}

        self.device = get_default_device()
        self.data, self.labels = self.read_data()
        self.model_layer_config = self.get_layer_config()

    def get_layer_config(self):
        if self.layer_config_file is None:
            return None

        session = botocore.session.get_session()
        s3 = session.create_client('s3', region_name=AWS_REGION)
        obj = s3.Object(S3_BUCKET, self.layer_config_file)
        config = json.load(obj.get()['Body'])

        return config

    def connect_visom(self):
        if self.visdom_config is None:
            return None

        return visdom.Visdom(
            server=self.visdom_config.url,
            port=self.visdom_config.port,
            username=self.visdom_config.username,
            password=self.visdom_config.password,
            use_incoming_socket=False,
            env=self.visdom_config.env
        )

    def split_data(self):
        begin = timer()
        split_selector = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.train_size,
            random_state=self.random_state
        )

        train_data = None
        val_data = None
        train_labels = None
        val_labels = None
        for train_index, val_index in split_selector.split(self.data, self.labels):
            train_data = self.data.iloc[train_index]
            val_data = self.data.iloc[val_index]
            train_labels = self.labels[train_index]
            val_labels = self.labels[val_index]

        self.run_metrics['split_data_time'] = timer() - begin
        return train_data, train_labels, val_data, val_labels

    def read_data(self):
        begin = timer()
        df = pd.read_csv(self.train_file, index_col="Index")

        labels = df.LABEL.values
        df = df.drop(columns=['LABEL'])

        self.run_metrics['read_data_time'] = timer() - begin
        return df, labels

    def prep_loaders(self):
        train_data, train_labels, val_data, val_labels = self.split_data()

        begin = timer()
        sparsifier = DLSparseMaker(self.num_symptoms)
        sparsifier.fit(train_data)

        self.age_std = sparsifier.age_std
        self.age_mean = sparsifier.age_mean

        train_data = sparsifier.transform(train_data)
        val_data = sparsifier.transform(val_data)

        input_dim = train_data.shape[1]
        assert input_dim == self.input_dim, \
            "Dimension of prepped data (%d) does not match specified input dimension (%d)" % (input_dim, self.input_dim)

        self.run_metrics['sparsify_data_time'] = timer() - begin

        begin = timer()
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
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        train_loader = DeviceDataLoader(train_loader, self.device)
        val_loader = DeviceDataLoader(val_loader, self.device)

        self.run_metrics['data_loader_time'] = timer() - begin

        return train_loader, val_loader

    def compose_runner(self):

        train_loader, val_loader = self.prep_loaders()
        model = self.compose_model()
        visdom = self.connect_visom()

        if visdom is None:
            print("Say whaaa!")

        return Runner(model, train_loader, val_loader, visdom=visdom, epochs=self.epochs)

    def compose_model(self):
        begin = timer()
        model = DNN(
            self.input_dim,
            self.num_conditions,
            layer_config=self.model_layer_config,
            non_linearity=self.non_linearity
        )

        model = to_device(model, self.device)

        self.run_metrics['model_composition_time'] = timer() - begin

        return model

    def run(self):
        # compose the runner
        self.run_metrics = {}
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.run_name)
        with mlflow.start_run():
            self.runner = self.compose_runner()

            # fit
            begin = timer()
            self.runner.fit()
            self.run_metrics['fit_time'] = timer() - begin

            # save results
            self.save_results()

            mlflow.log_metrics(self.run_metrics)

    def save_results(self):
        if os.path.exists(self.runner.early_stopping.path):
            model_dict = torch.load(self.runner.early_stopping.path)
        else:
            model_dict = self.runner.model.state_dict()

        epoch_count = self.runner.get_epoch()
        self.run_metrics['epochs'] = epoch_count
        self.run_metrics['train_accuracy_score'] = self.runner.train_loss[epoch_count-1]
        self.run_metrics['test_accuracy_score'] = self.runner.val_loss[epoch_count - 1]

        data = {
            "input_dim": self.input_dim,
            "output_dim": self.num_conditions,
            "layer_config": self.model_layer_config,
            "train_acc": self.runner.train_acc,
            "val_acc": self.runner.val_acc,
            "train_loss": self.runner.train_loss,
            "val_loss": self.runner.val_loss,
            "epoch": epoch_count,
            "model_dict": model_dict,
            "age_std": self.age_std,
            "age_mean": self.age_mean
        }

        filename = "%d.torch" % int(math.ceil(time.time()))
        tmp_path = os.path.join(self.tmp_directory, filename)
        s3_filename = self.run_name + "/" + filename

        torch.save(data, tmp_path)
        mlflow.log_artifact(tmp_path, s3_filename)
        return True


def train_dl(
        run_name,
        train_file,
        mlflow_uri,
        input_dim,
        num_sympoms,
        num_conditions,
        visdom_url,
        visdom_port,
        visdom_username,
        visdom_password,
        visdom_env,
        layer_config_file,
        **kwargs
):

    visdom_config = VisdomConfig
    visdom_config.url = visdom_url
    visdom_config.port = visdom_port
    visdom_config.username = visdom_username
    visdom_config.password = visdom_password
    visdom_config.env = visdom_env

    bench = Bench(
        run_name,
        train_file,
        mlflow_uri,
        input_dim,
        num_sympoms,
        num_conditions,
        visdom_config=visdom_config,
        layer_config=layer_config_file,
        **kwargs
    )

    bench.run()
