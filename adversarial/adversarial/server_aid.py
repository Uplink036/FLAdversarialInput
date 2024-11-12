
from functools import partial, reduce
from typing import Optional, Union
from flwr.common import FitIns, Parameters, FitRes, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, NDArray
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, Krum
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
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Compute in-place weighted average."""
        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)
        print(f"Total examples: {num_examples_total}")
        # Compute scaling factors for each result
        scaling_factors = np.asarray(
            [fit_res.num_examples / num_examples_total for _, fit_res in results]
        )

        def _try_inplace(
            x: NDArray, y: Union[NDArray, np.float64], np_binary_op: np.ufunc
        ) -> NDArray:
            return (  # type: ignore[no-any-return]
                np_binary_op(x, y, out=x)
                if np.can_cast(y, x.dtype, casting="same_kind")
                else np_binary_op(x, np.array(y, x.dtype), out=x)
            )

        # Let's do in-place aggregation
        # Get first result, then add up each other
        params = [
            _try_inplace(x, scaling_factors[0], np_binary_op=np.multiply)
            for x in parameters_to_ndarrays(results[0][1].parameters)
        ]

        for i, (_, fit_res) in enumerate(results[1:], start=1):
            res = (
                _try_inplace(x, scaling_factors[i], np_binary_op=np.multiply)
                for x in parameters_to_ndarrays(fit_res.parameters)
            )
            params = [
                reduce(partial(_try_inplace, np_binary_op=np.add), layer_updates)
                for layer_updates in zip(params, res)
            ]

        return params

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

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) :
        if (config.get_defense_number() == 1):
            # Get fitresults
            fit_results = [fit_res for (_, fit_res) in results]
            # Get train_loss from each client
            train_losses = [fit_res.metrics["train_loss"] for fit_res in fit_results]
            print(f"Train losses: {train_losses}")
            # Remove the client with the highest train loss
            max_loss_idx = np.argmax(train_losses)
            print(f"Client with highest loss: {max_loss_idx}")
            results.pop(max_loss_idx)
            
            fit_results = [fit_res for (_, fit_res) in results]
            # Get train_loss from each client
            train_losses = [fit_res.metrics["train_loss"] for fit_res in fit_results]
            print(f"Train losses: {train_losses}")
        elif (config.get_defense_number() == 2):
            pass
        return super().aggregate_fit(server_round, results, failures)
    
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
    
class CustomClientConfigStrategyKrum(Krum):
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