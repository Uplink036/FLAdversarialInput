"""Adversarial: A Flower / PyTorch app."""
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from adversarial.task import Net, get_weights
import pandas as pd
import os
import dotenv
from adversarial.configs import Configs
from adversarial.server_aid import weighted_average, CustomClientConfigStrategyFedAvg, CustomClientConfigStrategyFedProx, fit_config

def server_fn(context: Context):
    # Read from config
    dotenv.load_dotenv()

    count = int(os.environ['COUNT'].strip())

    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    config = Configs("adversarial/configs.json", count)
    # Define strategy

    if config.get_aggregate_fn() == "fedavg":
        strategy = CustomClientConfigStrategyFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
        )
    elif config.get_aggregate_fn() == "fedprox":
        strategy = CustomClientConfigStrategyFedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
            proximal_mu=config.get_proximal_mu(),
        )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
