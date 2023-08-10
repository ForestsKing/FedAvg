import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

from data.dataset import MnistDataset
from federated.client import Client
from federated.server import Server


class FedAvg:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting

        self.device = self._acquire_device()
        self.data_path, self.log_path, self.result_path = self._make_dirs()
        local_train_dataloaders, test_dataloader = self._create_mnist_dataloader()
        self.writer = SummaryWriter(log_dir=self.log_path)

        self.server = Server(self.args, test_dataloader, self.device)
        self.clients = [
            Client(self.args, k, local_train_dataloaders[k], self.device) for k in range(self.args.num_clients)
        ]

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            device = torch.device('cuda:{}'.format(self.args.devices))
            print('Use GPU: cuda:{}'.format(self.args.devices))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _make_dirs(self):
        data_path = self.args.data_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        log_path = os.path.join(self.args.save_path + '/log/', self.setting)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        result_path = os.path.join(self.args.save_path + '/result/', self.setting)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        return data_path, log_path, result_path

    def _create_mnist_dataloader(self):
        train_dataset = datasets.MNIST(root=self.args.data_path, download=self.args.download, train=True)
        test_dataset = datasets.MNIST(root=self.args.data_path, download=self.args.download, train=False)

        train_image = train_dataset.data.numpy()
        train_label = train_dataset.targets.numpy()
        test_image = test_dataset.data.numpy()
        test_label = test_dataset.targets.numpy()

        train_sorted_index = np.argsort(train_label)
        train_image = train_image[train_sorted_index]
        train_label = train_label[train_sorted_index]

        shard_start_index = [i for i in range(0, len(train_image), self.args.shard_size)]
        random.shuffle(shard_start_index)
        num_shards = len(shard_start_index) // self.args.num_clients

        local_train_dataloaders = []
        for client_id in range(self.args.num_clients):
            _index = num_shards * client_id
            images, labels = [], []
            for i in range(num_shards):
                images.append(
                    train_image[shard_start_index[_index + i]:shard_start_index[_index + i] + self.args.shard_size])
                labels.append(
                    train_label[shard_start_index[_index + i]:shard_start_index[_index + i] + self.args.shard_size])
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            local_train_dataset = MnistDataset(client_id, images, labels)
            local_train_dataloader = DataLoader(local_train_dataset, batch_size=self.args.batch_size, shuffle=True)
            local_train_dataloaders.append(local_train_dataloader)

        test_dataset = MnistDataset(-1, test_image, test_label)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        return local_train_dataloaders, test_dataloader

    def _train_step(self):
        for client in self.clients:
            client.set_model(self.server.get_model())

        n_sample = max(int(self.args.fraction * self.args.num_clients), 1)
        sample_set = np.random.randint(0, self.args.num_clients, n_sample)

        sample_clients = []
        for k in sample_set:
            self.clients[k].local_update()
            sample_clients.append(self.clients[k])

        total_data_size = sum([len(client) for client in sample_clients])
        weights = [len(client) / total_data_size for client in sample_clients]

        self.server.aggregation(sample_clients, weights)

    def _validation_step(self):
        test_loss, test_acc = self.server.validation()
        return test_loss, test_acc

    def fit(self):
        result = pd.DataFrame(columns=['round', 'test_loss', 'test_acc'])
        for t in range(self.args.comm_round):
            self._train_step()
            test_loss, test_acc = self._validation_step()

            print("Round: {0} || Test Loss: {1:.6f} Test ACC: {2:.4f}".format(t, test_loss, test_acc))

            self.writer.add_scalar('train/test_loss', test_loss, t)
            self.writer.add_scalar('train/test_acc', test_acc, t)

            result.loc[len(result)] = [t, test_loss, test_acc]

        self.writer.close()
        result.to_csv(self.result_path + '/result.csv', index=False)
