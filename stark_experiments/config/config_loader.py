import yaml
import os

def load_hyperparameters(config_path="hyperparameters.yaml", mode="default"):
    """
    Loads the YAML file and returns the specific configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if mode not in full_config:
        raise ValueError(f"Mode '{mode}' not found in {config_path}. Available: {list(full_config.keys())}")
    
    config = full_config[mode]
    
    # Remove keys that aren't arguments for retrieve() to prevent TypeErrors
    # (e.g., 'base_pheromone' isn't currently an argument in your retrieve function)
    clean_config = {k: v for k, v in config.items() if k != 'base_pheromone'}
    
    return clean_config