
from functools import partial, reduce
from typing import Optional, Union
from flwr.common import FitIns, Parameters, FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx
import pandas as pd
import numpy as np
from adversarial.configs import Configs
import os
import dotenv

dotenv.load_dotenv()
count = int(os.environ['COUNT'])
config = Configs("adversarial/configs.json", count)

counter = 0
def weighted_evaluate_average(metrics: list[tuple[int, dict[str, float]]]):
    global counter

    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples*m["accuracy"] for num_examples, m in metrics]
    kappa = [num_examples*m["kappa"] for num_examples, m in metrics]
    f1 = [num_examples*m["f1"] for num_examples, m in metrics]
    roc_auc = [num_examples*m["roc_auc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    print(metrics)
    n_examples = sum(examples)
    main_avg = {"accuracy": sum(accuracies) / n_examples,
                "kappa": sum(kappa) / n_examples,
                "f1": sum(f1) / n_examples,
                "roc_auc": sum(roc_auc) / n_examples}
    log = pd.DataFrame(main_avg, index=[0])
    log.loc[0] = [value for value in list(main_avg.values())]
    for i in range(len(metrics)):
        log.loc[i+1] = {key: value for key, value in metrics[i][1].items()}

    counter += 1
    log.to_csv(f"adversarial/logging/{count}/{counter}_log.csv")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples) , 
            "accuracy_per_client": [m["accuracy"] for _, m in metrics],
            "kappa": sum(kappa) / sum(examples), 
            "kappa_per_client": [m["kappa"] for _, m in metrics]}


def outlier_fit_average(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        num_examples_total = sum(num_examples for (_, num_examples) in results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        
        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        aggregated_ndarrays = weights_prime(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {"malicious": False}
    return config

class CustomClientConfigStrategyFedAvg(FedAvg):
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(num_clients=sample_size)

        global config
        attack_targets = config.get_attacker_num()
        attack_config = {"malicious": True}
        
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < attack_targets:
                fit_configurations.append((client, FitIns(parameters, attack_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, {}))
                )
        return fit_configurations
        # else:
        #     return super().configure_fit(server_round, parameters, client_manager)

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

class CustomClientConfigStrategyFedProx(FedProx):
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(num_clients=sample_size)

        # if server_round == 1:
        global config
        attack_targets = config.get_attacker_num()
        attack_config = {"malicious": True}

        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < attack_targets:
                fit_configurations.append((client, FitIns(parameters, attack_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, {}))
                )
        return fit_configurations
        # else:
        #     return super().configure_fit(server_round, parameters, client_manager)

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients