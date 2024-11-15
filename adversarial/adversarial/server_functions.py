
from functools import partial, reduce
from typing import Optional, Union
from flwr.common import Parameters, FitRes, Scalar, parameters_to_ndarrays, NDArray
from flwr.server.client_proxy import ClientProxy
import pandas as pd
import numpy as np
import os
import dotenv
from adversarial.configs import Configs

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

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {"malicious": False}
    return config
