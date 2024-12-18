import dotenv
import os
import pandas as pd
from adversarial.configs import Configs

dotenv.load_dotenv()
count = int(os.environ['COUNT'])
config = Configs("adversarial/configs.json", count)

counter = 0
def weighted_evaluate_average(
        metrics: list[tuple[int, dict[str, float]]]
    ):
    # Multiply accuracy of each client by number of examples used
    metric_length = len(metrics)
    examples    = [0]*metric_length
    accuracies  = [0]*metric_length
    kappa       = [0]*metric_length
    f1          = [0]*metric_length
    roc_auc     = [0]*metric_length
    
    i = 0
    for num_examples, m in metrics:
        accuracies[i]  = num_examples*m["accuracy"]
        kappa[i]       = num_examples*m["kappa"]
        f1[i]          = num_examples*m["f1"]
        roc_auc[i]     = num_examples*m["roc_auc"]
        examples[i]    = num_examples
        i += 1

    n_examples  = sum(examples)

    main_avg = {"accuracy": sum(accuracies) / n_examples,
                "kappa":    sum(kappa) / n_examples,
                "f1":       sum(f1) / n_examples,
                "roc_auc":  sum(roc_auc) / n_examples}
    
    log = pd.DataFrame(main_avg, index=[0])
    log.loc[0] = [value for value in list(main_avg.values())]
    for i in range(len(metrics)):
        log.loc[i+1] = {key: value for key, value in metrics[i][1].items()}

    global counter
    counter += 1
    log.to_csv(f"adversarial/logging/{count}/{counter}_log.csv")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples) , 
            "accuracy_per_client": [m["accuracy"] for _, m in metrics],
            "kappa": sum(kappa) / sum(examples), 
            "kappa_per_client": [m["kappa"] for _, m in metrics]}

def fit_config(
        server_round: int
    ):
    """Return training configuration dict for each round."""
    config = {"malicious": False}
    return config
