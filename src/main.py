import gymnasium
import numpy as np
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from simglucose.analysis.risk import risk_index  # type: ignore
from cmd_args import parse_args
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def custom_reward_fn(BG_last_hour):
    """Calculate the reward based on the risk index difference."""
    if len(BG_last_hour) < 2:
        return 0
    _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
    _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
    return risk_current - risk_prev


def register_env(cfg):
    """Register the custom environment using configuration settings."""
    register(
        id=cfg['env']['id'],
        entry_point=cfg['env']['entry_point'],
        max_episode_steps=cfg['env']['max_episode_steps'],
        kwargs={"patient_name": cfg['env']['patient_name'], 'reward_fun': custom_reward_fn},
    )


def make_env(cfg, render_mode=None):
    """Create and return a gym environment wrapped with Monitor."""
    env = gymnasium.make(cfg['env']['id'], render_mode=render_mode, seed=cfg["seed"])
    log_dir = Path(cfg.get("monitor_log_dir", "monitor_logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, filename=str(log_dir))
    return env


def select_device(cfg):
    """Select device based on availability and configuration."""
    # Auto-detect if not provided in the config
    if 'device' in cfg and cfg['device']:
        device = cfg['device']
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    # Update the config so that downstream functions use the correct device
    cfg['device'] = device
    return device


def train(cfg):
    """Training routine for the DDPG agent."""
    logger.info("Starting training...")
    env_config = cfg["env"]
    env = make_env(cfg, render_mode=None)

    n_actions = env.action_space.shape[-1]
    sigma = cfg["action_noise"]["sigma"]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))

    model_config = cfg["model"]
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        device=cfg["device"],
        learning_rate=model_config["learning_rate"],
        buffer_size=model_config["buffer_size"],
        batch_size=model_config["batch_size"],
        gamma=model_config["gamma"],
        tensorboard_log=cfg["training"]["tensorboard_log"],
    )

    # Set up evaluation and checkpoint callbacks
    eval_env = make_env(cfg, render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg["training"]["log_path"],
        log_path=cfg["training"]["log_path"],
        eval_freq=cfg["eval"]["eval_freq"],
        n_eval_episodes=cfg["eval"]["n_eval_episodes"],
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["training"]["checkpoint_freq"],
        save_path=cfg["training"]["save_path"],
        name_prefix='ddpg_checkpoint',
    )

    # Start training
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=[eval_callback, checkpoint_callback]
    )

    # Save the trained model
    model_path = cfg.get("model_save_path", "ddpg_simglucose")
    model.save(model_path)
    logger.info(f"Model saved as '{model_path}'.")


def predict(cfg):
    """Prediction/inference routine for the trained DDPG agent."""
    logger.info("Starting prediction...")
    env = make_env(cfg, render_mode="human")
    try:
        model = DDPG.load(cfg.get("model_save_path", "ddpg_simglucose"), device=cfg["device"])
    except Exception as e:
        logger.error(f"Error loading model with model_save_path: {e}")
        return

    logger.info(f"Model loaded from '{cfg.get('model_save_path', 'ddpg_simglucose')}'.")
    observation, info = env.reset(seed=cfg["seed"])

    max_steps = cfg.get("predict_steps", 200)
    for t in range(max_steps):
        env.render()
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        logger.info(
            f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
        )
        if terminated or truncated:
            logger.info("Episode finished after {} timesteps".format(t + 1))
            break


def main():
    # Parse configuration from YAML
    cfg = parse_args()  
    # Set a fixed seed for reproducibility
    np.random.seed(cfg['seed'])

    # Select device and update the configuration
    select_device(cfg)

    # Register the custom environment
    register_env(cfg)

    # Decide on the mode (train or predict) based on configuration
    mode = cfg.get("mode", "train").lower()
    print(mode)
    if mode == "train":
        train(cfg)
    elif mode == "predict":
        predict(cfg)
    else:
        logger.error(f"Unknown mode '{mode}'. Please choose 'train' or 'predict'.")


if __name__ == "__main__":
    main()
