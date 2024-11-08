import json

class Configs():
    def __init__(self, configs_file, config_num):
        self._config_num = config_num
        self.configs = json.load(open(configs_file))["configs"]
        self._proximal_mu = 1.0
        self._classes_per_partition = 7
        self._partition_by = "label"
    
    def _get_current_config(self):
        return self.configs[self._config_num]
    
    def get_distribution(self):
        return self._get_current_config()["dist"]
    
    def get_attacker_num(self):
        return self._get_current_config()["n_attack"]
    
    def get_aggregate_fn(self):
        return self._get_current_config()["aggregate"]

    def get_proximal_mu(self):
        if self.get_aggregate_fn() == "fedprox":
            return self._proximal_mu
        else:
            raise ValueError("Proximal mu is only used in FedProx")
    
    def get_classes_per_partition(self):
        if self.get_distribution() == "noniid":
            return self._classes_per_partition
        else:
            raise ValueError("Classes per partition is only used in Non-IID")
    
    def get_partition_by(self):
        if self.get_distribution() == "noniid":
            return self._partition_by
        else:
            raise ValueError("Partition by is only used in Non-IID")
        
    def get_defense_number(self):
        return self._get_current_config()["defense"]
        
    


