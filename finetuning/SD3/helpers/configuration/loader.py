import os
import logging
import yaml                                # ← ensure this import is present
from helpers.configuration import toml_file, json_file, env_file, cmd_args
from helpers.training.state_tracker import StateTracker
import sys

def load_yaml_config():
    """
    Load a YAML config at $CONFIG_PATH.yaml, returning the parsed dict.
    """
    path = os.environ.get("CONFIG_PATH", "config/config") + ".yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)

logger = logging.getLogger("SimpleTuner")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))

helpers = {
    "json": json_file.load_json_config,
    "toml": toml_file.load_toml_config,
    "env":  env_file.load_env_config,
    # "yaml": load_yaml_config,              # ← and this entry
    "cmd":  cmd_args.parse_cmdline_args,
}

default_config_paths = {
    "json": "config.json",
    "toml": "config.toml",
    "env":  "config.env",
    # "yaml": "config.yaml",
}

def attach_env_to_path_if_not_present(backend: str, env: str = None):
    backend_cfg_path = default_config_paths.get(backend)
    if env and env != "default":
        return f"config/{env}/{backend_cfg_path}"
    return f"config/{backend_cfg_path}"

def load_config(args: dict = None, exit_on_error: bool = False):
    # Bypass if the user just wants help
    if "-h" in sys.argv or "--help" in sys.argv:
        return helpers["cmd"]()

    mapped_config = args
    if not mapped_config:
        config_backend = os.environ.get(
            "SIMPLETUNER_CONFIG_BACKEND",
            os.environ.get("CONFIG_BACKEND", os.environ.get("CONFIG_TYPE", "env")),
        ).lower()
        config_env = os.environ.get(
            "SIMPLETUNER_ENVIRONMENT",
            os.environ.get("SIMPLETUNER_ENV", os.environ.get("ENV", "default")),
        )
        config_backend_path = "config"
        if config_env and config_env != "default":
            config_backend_path = os.path.join("config", config_env)
        StateTracker.set_config_path(config_backend_path)

        logger.info(f"Using {config_backend} configuration backend.")
        mapped_config = helpers[config_backend]()
        if config_backend == "cmd":
            return mapped_config

    # Finally feed through argparse for defaults & validation
    configuration = helpers["cmd"](
        input_args=mapped_config, exit_on_error=exit_on_error
    )
    return configuration
