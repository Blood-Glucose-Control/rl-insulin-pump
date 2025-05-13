from src.agents.agent_loader import make_model
from src.environments.env_loader import make_env
import logging
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_network_config(n_layers, hidden_units):
    return {"n_layers": n_layers, "hidden_units": hidden_units, "activation_fn": "relu"}


def evaluate_network(cfg, network_config):
    """Evaluate a specific network architecture."""
    logger.info(
        f"Evaluating network with {network_config['n_layers']} layers and {network_config['hidden_units']} units"
    )

    # Create environment
    env = make_env(cfg, render_mode=None)
    eval_env = make_env(cfg, render_mode=None)

    # Create model with custom network architecture
    model = make_model(cfg, env, network_config)

    # Train and evaluate
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg["eval"]["n_eval_episodes"]
    )

    return {
        "config": network_config,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
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
                "policy_kwargs": {"net_arch": [hidden_units] * n_layers},
            }

            metrics = evaluate_network(cfg, network_config)
            layer_results.append(metrics)

            logger.info(f"Results for {n_layers} layers, {hidden_units} units:")
            logger.info(
                f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
            )

        # Find best configuration for this number of layers
        best_result = max(layer_results, key=lambda x: x["mean_reward"])
        results.append(best_result)

    return results


def plot_results(results):
    """Plot results from grid search"""

    n_layers = [r["config"]["n_layers"] for r in results]
    rewards = [r["mean_reward"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(n_layers, rewards, marker="o")
    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Reward")
    plt.title("Performance vs Network Depth")
    plt.grid(True)
    plt.savefig("network_performance.png")
    plt.close()

    # Create results table
    print("\nResults Table:")
    print("Layers | Hidden Units | Mean Reward | Std Reward")
    print("-" * 50)
    for r in results:
        print(
            f"{r['config']['n_layers']:6d} | {r['config']['hidden_units']:11d} | {r['mean_reward']:11.2f} | {r['std_reward']:10.2f}"
        )


def visualize_results(results, save_path="./results"):
    """Visualize grid search results."""
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Plot results
    n_layers = [r["config"]["n_layers"] for r in results]
    mean_rewards = [r["mean_reward"] for r in results]
    std_rewards = [r["std_reward"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(n_layers, mean_rewards, yerr=std_rewards, marker="o")
    plt.xlabel("Number of Layers")
    plt.ylabel("Mean Reward")
    plt.title("Performance vs Network Depth")
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
            f.write(
                f"{r['config']['n_layers']:6d} | {r['config']['hidden_units']:11d} | {r['mean_reward']:8.2f} ± {r['std_reward']:5.2f}\n"
            )
