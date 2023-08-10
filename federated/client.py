from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from model.model import Model


class Client:
    def __init__(self, args, client_id, dataloader, device):
        self.args = args
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device

        self.model = Model().to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)
        self.loss_fn = CrossEntropyLoss()

    def __len__(self):
        return len(self.dataloader.dataset)

    def set_model(self, model):
        global_state = model.state_dict()
        self.model.load_state_dict(global_state)

    def local_update(self):
        self.model.train()
        for e in range(self.args.local_epoch):
            for (batch_x, batch_y) in self.dataloader:
                self.optimizer.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()
