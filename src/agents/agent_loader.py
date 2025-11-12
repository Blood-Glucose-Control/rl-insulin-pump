import numpy as np
from stable_baselines3 import DDPG, PPO, DQN
from stable_baselines3.common.noise import NormalActionNoise
import logging
from src.extractors.lstm import LSTMExtractor

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

    network_config = cfg.get("policy_network_arch", None)


    if(model_config["policy"] == "LSTM"):
        policy_kwargs = dict(
            normalize_images=False,
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs=dict(
                hidden_dim=network_config["hidden_units"],
                layer_dim=network_config["n_layers"],
            ) if network_config else dict(hidden_dim=128, layer_dim=1),
        )
    else:
        policy_kwargs = (
            dict(net_arch=[network_config["hidden_units"]] * network_config["n_layers"], normalize_images=False)
            #  "discrete_action_space": model_config.get("discrete_action_space", False), "discrete_observation_space": model_config.get("discrete_observation_space", False)}
            if network_config
            else dict(normalize_images=False)
            #  "discrete_action_space": model_config.get("discrete_action_space", False), "discrete_observation_space": model_config.get("discrete_observation_space", False)}
        )

    policy = model_config["policy"] if model_config["policy"] != "LSTM" else "MlpPolicy"


    model_kwargs = {
        "policy": policy,
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
                "policy": policy,
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
        # TODO: This is made for working on Linux systems, Windows may need to add .zip to the file name
        case "DDPG":
            return DDPG.load(
                cfg.get("model_save_path", "ddpg_simglucose"), device=cfg["device"]
            )
        case "PPO":
            return PPO.load(
                cfg.get("model_save_path", "ppo_simglucose"), device=cfg["device"]
            )
        case "DQN":
            return DQN.load(
                cfg.get("model_save_path", "dqn_simglucose"), device=cfg["device"]
            )
        case _:
            raise ValueError(f"Unknown model name: {cfg['model_name']}")
