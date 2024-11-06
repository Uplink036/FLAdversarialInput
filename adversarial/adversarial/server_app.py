"""Adversarial: A Flower / PyTorch app."""
from flwr.common import Context, ndarrays_to_parameters, FitIns, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from adversarial.task import Net, get_weights

def weighted_average(metrics: list[tuple[int, dict[str, float]]]):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples*m["accuracy"] for num_examples, m in metrics]
    kappa = [num_examples*m["kappa"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

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

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = CustomClientConfigStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config        
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
