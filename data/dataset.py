from torch.utils.data import Dataset
from torchvision import transforms


class MnistDataset(Dataset):
    def __init__(self, client_id, images, labels):
        self.client_id = client_id
        self.images = images
        self.labels = labels.astype(int)

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, index):
        img = self.transform(self.images[index])
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
