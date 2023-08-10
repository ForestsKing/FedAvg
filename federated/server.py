import copy
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from model.model import Model


class Server():
    def __init__(self, args, dataloader, device):
        self.args = args
        self.dataloader = dataloader
        self.device = device

        self.model = Model().to(self.device)
        self.loss_fn = CrossEntropyLoss()

    def get_model(self):
        return copy.deepcopy(self.model)

    def aggregation(self, clients, weights):
        update_state = OrderedDict()

        for k, client in enumerate(clients):
            local_state = client.model.state_dict()
            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] * weights[k]
                else:
                    update_state[key] += local_state[key] * weights[k]

        self.model.load_state_dict(update_state)

    def validation(self):
        test_loss, test_label, test_pred = [], [], []

        with torch.no_grad():
            self.model.eval()
            for (batch_x, batch_y) in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)

                test_loss.append(loss.item())
                test_label.append(batch_y.cpu().numpy())
                test_pred.append(output.cpu().numpy())

        test_loss = np.mean(test_loss)
        test_label = np.concatenate(test_label, axis=0)
        test_pred = np.concatenate(test_pred, axis=0)
        test_pred = np.argmax(test_pred, axis=1)
        test_acc = np.sum(test_pred == test_label) / len(test_label)

        return test_loss, test_acc
