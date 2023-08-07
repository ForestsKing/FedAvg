from torchvision import datasets
from torchvision import transforms


def MyDataset(data_path, train, download):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_path, download=download, train=train, transform=transform)
    return dataset
