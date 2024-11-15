
import dotenv
import numpy as np
import os
from adversarial.configs import Configs
from flwr.common import FitIns, Parameters, FitRes, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx, Krum
from typing import Union

dotenv.load_dotenv()
count = int(os.environ['COUNT'])
config = Configs("adversarial/configs.json", count)

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