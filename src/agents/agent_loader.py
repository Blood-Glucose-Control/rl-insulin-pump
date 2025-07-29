import numpy as np
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_model(cfg, env, network_config=None):
    n_actions = env.action_space.shape[-1]
    sigma = cfg["action_noise"]["sigma"]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions)
    )

    model_config = cfg["model"]

    policy_kwargs = (
        {"net_arch": [network_config["hidden_units"]] * network_config["n_layers"]}
        if network_config
        else None
    )

    model_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "action_noise": action_noise,
        "verbose": 1,
        "device": cfg["device"],
        "learning_rate": model_config["learning_rate"],
        "buffer_size": model_config["buffer_size"],
        "batch_size": model_config["batch_size"],
        "gamma": model_config["gamma"],
        "tensorboard_log": model_config["run_directory"]+"/tensorboard/",
        "policy_kwargs": policy_kwargs,
    }

    match cfg["model_name"]:
        case "DDPG":
            return DDPG(**model_kwargs)
        case "PPO":
            return PPO(**model_kwargs)
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
