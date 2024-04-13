import yaml
import ast


def read_config(file_path: str) -> dict:
    """Read a YAML configuration file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Configuration parameters.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    # Evaluate values enclosed in brackets
    for key, value in config.items():
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            config[key] = ast.literal_eval(value)
    return config
