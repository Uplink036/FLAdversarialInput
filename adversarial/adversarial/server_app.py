"""Adversarial: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from adversarial.task import Net, get_weights
import pandas as pd

counter = 0
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

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
