import gymnasium
import numpy as np
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from simglucose.analysis.risk import risk_index  # type: ignore
from cmd_args import parse_args
import logging
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

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
    """Register the custom environment with network configuration."""
    kwargs = {
        "patient_name": cfg['env']['patient_name'],
        "reward_fun": custom_reward_fn
    }
    
    register(
        id=cfg['env']['id'],
        entry_point=cfg['env']['entry_point'],
        max_episode_steps=cfg['env']['max_episode_steps'],
        kwargs=kwargs
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
    """Training routine for the PPO agent."""
    logger.info("Starting training...")
    env_config = cfg["env"]
    env = make_env(cfg, render_mode=None)

    n_actions = env.action_space.shape[-1]
    sigma = cfg["action_noise"]["sigma"]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions))

    model_config = cfg["model"]
    model = PPO(
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
        name_prefix='ppo_checkpoint',
    )

    # Start training
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=[eval_callback, checkpoint_callback]
    )

    # Save the trained model
    model_path = cfg.get("model_save_path", "ppo_simglucose")
    model.save(model_path)
    logger.info(f"Model saved as '{model_path}'.")


def predict(cfg):
    """Prediction/inference routine for the trained PPO agent."""
    logger.info("Starting prediction...")
    env = make_env(cfg, render_mode="human")
    try:
        model = PPO.load(cfg.get("model_save_path", "ppo_simglucose"), device=cfg["device"])
    except Exception as e:
        logger.error(f"Error loading model with model_save_path: {e}")
        return

    logger.info(f"Model loaded from '{cfg.get('model_save_path', 'ppo_simglucose')}'.")
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


def create_network_config(n_layers, hidden_units):
    return {
        "n_layers": n_layers,
        "hidden_units": hidden_units,
        "activation_fn": "relu"
    }


def evaluate_network(cfg, network_config):
    """Evaluate a specific network architecture."""
    logger.info(f"Evaluating network with {network_config['n_layers']} layers and {network_config['hidden_units']} units")
    
    # Register environment without network config
    register_env(cfg)
    
    # Create environment
    env = make_env(cfg, render_mode=None)
    eval_env = make_env(cfg, render_mode=None)
    
    # Setup noise and model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), 
        sigma=cfg["action_noise"]["sigma"] * np.ones(n_actions)
    )
    
    # Create model with custom network architecture
    model_kwargs = cfg["model"].copy()  # Create a copy of model config
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=cfg["device"],
        policy_kwargs={"net_arch": [network_config["hidden_units"]] * network_config["n_layers"]},
        # **model_kwargs
    )
    
    # Train and evaluate
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=cfg["eval"]["n_eval_episodes"])
    
    return {
        "config": network_config,
        "mean_reward": mean_reward,
        "std_reward": std_reward
    }


def grid_search(cfg):
    """Run grid search over network configurations."""
    results = []
    
    # Define search space
    n_layers_range = range(1, 9)  # 1-8 layers
    hidden_units_range = [2**i for i in range(2, 9)]  # 4-256 units
    
    for n_layers in n_layers_range:
        layer_results = []
        for hidden_units in hidden_units_range:
            network_config = {
                "n_layers": n_layers,
                "hidden_units": hidden_units,
                "policy_kwargs": {
                    "net_arch": [hidden_units] * n_layers
                }
            }
            
            metrics = evaluate_network(cfg, network_config)
            layer_results.append(metrics)
            
            logger.info(f"Results for {n_layers} layers, {hidden_units} units:")
            logger.info(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        
        # Find best configuration for this number of layers
        best_result = max(layer_results, key=lambda x: x["mean_reward"])
        results.append(best_result)
    
    return results


def plot_results(results):
    """Plot results from grid search"""
    
    n_layers = [r["config"]["n_layers"] for r in results]
    rewards = [r["mean_reward"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_layers, rewards, marker='o')
    plt.xlabel('Number of Layers')
    plt.ylabel('Mean Reward')
    plt.title('Performance vs Network Depth')
    plt.grid(True)
    plt.savefig('network_performance.png')
    plt.close()
    
    # Create results table
    print("\nResults Table:")
    print("Layers | Hidden Units | Mean Reward | Std Reward")
    print("-" * 50)
    for r in results:
        print(f"{r['config']['n_layers']:6d} | {r['config']['hidden_units']:11d} | {r['mean_reward']:11.2f} | {r['std_reward']:10.2f}")


def visualize_results(results, save_path="./ppo-results"):
    """Visualize grid search results."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Plot results
    n_layers = [r["config"]["n_layers"] for r in results]
    mean_rewards = [r["mean_reward"] for r in results]
    std_rewards = [r["std_reward"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_layers, mean_rewards, yerr=std_rewards, marker='o')
    plt.xlabel('Number of Layers')
    plt.ylabel('Mean Reward')
    plt.title('Performance vs Network Depth')
    plt.grid(True)
    plt.savefig(f"{save_path}/network_performance.png")
    plt.close()
    
    # Save results table
    with open(f"{save_path}/results.txt", "w") as f:
        f.write("Network Architecture Results\n")
        f.write("-" * 50 + "\n")
        f.write("Layers | Hidden Units | Mean Reward ± Std\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"{r['config']['n_layers']:6d} | {r['config']['hidden_units']:11d} | {r['mean_reward']:8.2f} ± {r['std_reward']:5.2f}\n")


def main():
    # Parse configuration from YAML
    cfg = parse_args()  
    # Set a fixed seed for reproducibility
    np.random.seed(cfg['seed'])

    # Select device and update the configuration
    select_device(cfg)

    # Decide on the mode based on configuration
    mode = cfg.get("mode", "train").lower()
    
    if mode == "train":
        # Register the custom environment
        register_env(cfg)
        train(cfg)
    elif mode == "predict":
        # Register the custom environment
        register_env(cfg)
        predict(cfg)
    elif mode == "grid_search":
        # Run grid search over network configurations
        results = grid_search(cfg)
        # Visualize and save results
        visualize_results(results)
        # Also print results to console
        plot_results(results)
    else:
        logger.error(f"Unknown mode '{mode}'. Please choose 'train', 'predict', or 'grid_search'.")


if __name__ == "__main__":
    main()
