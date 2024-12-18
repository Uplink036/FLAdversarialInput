"""Adversarial: A Flower / PyTorch app."""

import dotenv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from adversarial.configs import Configs
from collections import OrderedDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset
def load_data(
        partition_id: int, 
        num_partitions: int
    ):
    """Load partition CIFAR10 data."""
    dotenv.load_dotenv()
    count = int(os.environ['COUNT'].strip())
    config = Configs("adversarial/configs.json", count)

    global fds
    if fds is None:
        if config.get_distribution() == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by=config.get_partition_by(), alpha=config.get_alpha())
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(
        net, 
        trainloader, 
        epochs, 
        device
    ):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(
        net, 
        testloader, 
        device
    ):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    y_test, y_pred = [], []
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            y_test += [label.item() for label in labels]
            _, y_pred = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (y_pred == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(testloader)
    f1, roc_auc, cohen_kappa_score = eval_metrics(labels, y_pred)
    return loss, accuracy, f1, roc_auc, cohen_kappa_score

def eval_metrics(
        y_true, 
        y_pred
    ):
    f1 = f1_score(y_true, y_pred, average='weighted')
    cohen_kappa = cohen_kappa_score(y_true, y_pred)

    y_pred = F.one_hot(y_pred, num_classes=10)
    y_true = y_true.numpy()
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo', labels=range(0,10))
  
    return f1, roc_auc, cohen_kappa

def get_weights(
        net
    ):
    weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return weights


def set_weights(
        net, 
        parameters
    ):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
