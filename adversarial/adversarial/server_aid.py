from flwr.common import FitIns, Parameters
from flwr.server.strategy import FedAvg

def weighted_average(metrics: list[tuple[int, dict[str, float]]]):
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
    log.to_csv("adversarial/logging/log" + str(counter) + ".csv")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples) , 
            "accuracy_per_client": [m["accuracy"] for _, m in metrics],
            "kappa": sum(kappa) / sum(examples), 
            "kappa_per_client": [m["kappa"] for _, m in metrics]}

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {"malicious": False}
    return config

class CustomClientConfigStrategy(FedAvg):
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(num_clients=sample_size)

        # if server_round == 1:
        attack_targets = 1
        attack_config = {"malicious": True}
        config = {"malicious": False}
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
