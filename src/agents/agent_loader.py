import numpy as np
from stable_baselines3 import DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_model(cfg, env, network_config=None):
    if cfg["env"].get("discrete_action_space", False):
        n_actions = env.action_space.n
    else:
        n_actions = env.action_space.shape[-1]
    sigma = cfg["action_noise"]["sigma"]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions)
    )

    model_config = cfg["model"]

    policy_kwargs = (
        {"net_arch": [network_config["hidden_units"]] * network_config["n_layers"]}
        #  "discrete_action_space": model_config.get("discrete_action_space", False), "discrete_observation_space": model_config.get("discrete_observation_space", False)}
        if network_config
        else None
        #  "discrete_action_space": model_config.get("discrete_action_space", False), "discrete_observation_space": model_config.get("discrete_observation_space", False)}
    )

    model_kwargs = {
        "policy": model_config["policy"],  # TODO: make this configurable see: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
        "env": env,
        "action_noise": action_noise,
        "verbose": 1,
        "device": cfg["device"],
        "learning_rate": model_config["learning_rate"],
        "buffer_size": model_config["buffer_size"],
        "batch_size": model_config["batch_size"],
        "gamma": model_config["gamma"],
        "tensorboard_log": cfg["run_directory"] + "/tensorboard/",
        "policy_kwargs": policy_kwargs,
    }

    match cfg["model_name"]:
        case "DDPG":
            return DDPG(**model_kwargs)
        case "PPO":
            return PPO(**model_kwargs)
        case "DQN":
            model_kwargs = {
                "policy": model_config["policy"],  # TODO: make this configurable see: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
                "env": env,
                "verbose": 1,
                "device": cfg["device"],
                "learning_rate": model_config["learning_rate"],
                "batch_size": model_config["batch_size"],
                "gamma": model_config["gamma"],
                "tensorboard_log": cfg["run_directory"] + "/tensorboard/",
                "policy_kwargs": policy_kwargs,
            }
            return DQN(**model_kwargs)
        case _:
            raise ValueError(f"Unknown model name: {cfg['model_name']}")


def load_model(cfg):
    match cfg["model_name"]:
        case "DDPG":
            return DDPG.load(
                cfg.get("model_save_path", "ddpg_simglucose"), device=cfg["device"]
            )
        case "PPO":
            return PPO.load(
                cfg.get("model_save_path", "ppo_simglucose"), device=cfg["device"]
            )
        case _:
            raise ValueError(f"Unknown model name: {cfg['model_name']}")
