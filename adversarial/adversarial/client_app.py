"""Adversarial: A Flower / PyTorch app."""

import numpy as np
import torch
from adversarial.task import Net, get_weights, load_data, set_weights, test, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, partition_id, trainloader, valloader, local_epochs):
        self.net = net
        self.partition_id = partition_id
        self.trainloader: DataLoader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(
            self, 
            parameters, 
            config
        ):
        if "malicious" in config.keys() and config["malicious"] == True:
            current_weights = get_weights(self.net)
            for layer in current_weights:
                np.random.shuffle(layer)
            set_weights(self.net, current_weights)
            config["malicious"] = False
        else:
            set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(
            self, 
            parameters, 
            config
        ):
        set_weights(self.net, parameters)
        loss, accuracy, f1, roc_auc, kappa = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy, "kappa": kappa, "f1": f1, "roc_auc": roc_auc}


def client_fn(
        context: Context
    ):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, partition_id, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
