import gymnasium
import numpy as np
import pandas as pd
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from simglucose.analysis.risk import risk_index  # type: ignore
from src.cmd_args import parse_args
import logging
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from gymnasium import Wrapper
from simglucose.analysis.report import report
from src.environment.patient import get_default_patients
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


class MultiPatientEnv(Wrapper):
    """Environment wrapper that cycles through multiple patients.

    This class allows training a single model on multiple different patients,
    cycling through them during training to ensure the model generalizes well.

    The environment handles:
    1. Creating unique instances for each patient
    2. Cycling through patients when episodes end
    3. Forcing patient switches after a set number of steps
    4. Maintaining proper Gymnasium environment interface
    """

    def __init__(
        self,
        patient_names,
        env_id,
        entry_point,
        max_episode_steps,
        reward_fun,
        seed=42,
        render_mode=None,
    ):
        """Initialize with a list of patient names.

        Args:
            patient_names: List of patient identifiers (e.g., ["adolescent#001", "adult#001"])
            env_id: Base environment ID (e.g., "simglucose/multi-patient-v0")
            entry_point: Python path to the environment class
            max_episode_steps: Maximum steps per episode
            reward_fun: Reward function to use
            seed: Random seed for reproducibility
            render_mode: Visualization mode (if any)
        """
        self.patient_names = patient_names
        self.current_patient_idx = 0  # Start with the first patient
        self.base_env_id = env_id
        self.entry_point = entry_point
        self.max_episode_steps = max_episode_steps
        self.reward_fun = reward_fun
        self.seed = seed
        self._render_mode = (
            render_mode  # Use _render_mode to avoid conflicts with Wrapper class
        )
        self.steps_with_current_patient = 0  # Track steps with current patient
        self.patient_switch_interval = max_episode_steps  # Switch after this many steps

        # Register and create the first environment
        self._register_current_env()
        env = gymnasium.make(
            self.current_env_id, render_mode=self._render_mode, seed=self.seed
        )
        super().__init__(
            env
        )  # Initialize the Wrapper with the first patient's environment

        logger.info(f"Initialized MultiPatientEnv with {len(patient_names)} patients")

    def _register_current_env(self):
        """Register the environment with the current patient.

        This method:
        1. Creates a valid Gymnasium ID for the current patient
        2. Registers the environment with Gymnasium
        3. Sets up the environment configuration for this patient

        The registration process ensures each patient has a unique environment ID
        that follows Gymnasium's naming conventions.
        """
        patient_name = self.patient_names[self.current_patient_idx]

        # Convert patient name to be valid in an environment ID (replace invalid chars)
        # For example: "adolescent#001" -> "adolescent_001"
        safe_patient_id = patient_name.replace("#", "_").replace("/", "_")

        # The only requirement for Gymnasium environment IDs is global uniqueness.
        # We control the format of `base_env_id` via the config, so no complex parsing is needed.
        # By appending a sanitized `safe_patient_id` to the `base_env_id`, we ensure each environment
        # has a unique, predictable ID for Gym registration (e.g., "simglucose/multi-patient-v0-adolescent_001").
        # This simple pattern meets Gym's requirements while keeping the code clean.

        self.current_env_id = f"{self.base_env_id}-{safe_patient_id}"

        # Set up the environment-specific parameters
        kwargs = {
            "patient_name": patient_name,  # The actual patient identifier
            "reward_fun": self.reward_fun,  # The reward function to use
        }

        try:
            # Register with Gymnasium - this makes the environment available for creation
            register(
                id=self.current_env_id,
                entry_point=self.entry_point,
                max_episode_steps=self.max_episode_steps,
                kwargs=kwargs,
            )
            logger.info(
                f"Registered environment for patient: {patient_name} with ID: {self.current_env_id}"
            )
        except gymnasium.error.Error:
            # If already registered, just use the existing registration
            logger.info(
                f"Using existing environment for patient: {patient_name} with ID: {self.current_env_id}"
            )

    def step(self, action):
        """Execute one step in the environment.

        This method:
        1. Executes the action in the underlying environment
        2. Tracks steps with the current patient
        3. Forces patient switching if we've spent too long with one patient
        4. Handles episode termination signaling

        Forcing termination after a certain number of steps ensures the
        training algorithm sees a variety of patients, not just the easiest ones.

        Args:
            action: The action to take in the environment

        Returns:
            observation: The new observation
            reward: The reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated (e.g., max steps)
            info: Additional information
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Increment step counter for the current patient
        self.steps_with_current_patient += 1

        # Force switch to next patient after certain number of steps
        # This ensures we don't spend too much time on any single patient
        if self.steps_with_current_patient >= self.patient_switch_interval:
            logger.info(
                f"Forcing patient switch after {self.steps_with_current_patient} steps"
            )
            # Mark episode as done to force reset
            terminated = True

            # Add info about patient switch for monitoring
            if isinstance(info, dict):
                info["patient_switch"] = True

        # If episode is done, log and prepare for patient switch
        if terminated or truncated:
            logger.info(
                f"Episode finished for patient: {self.patient_names[self.current_patient_idx]}"
            )
            # The actual patient switch will happen in reset()

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and switch to the next patient.

        This method is called:
        1. At the start of training
        2. When an episode ends (termination or truncation)
        3. When explicitly called by the training algorithm

        It cycles through the patients in round-robin fashion, ensuring
        all patients are used during training.

        Args:
            **kwargs: Additional arguments passed to the environment reset

        Returns:
            The initial observation from the new environment
        """
        # Reset step counter for the new patient
        self.steps_with_current_patient = 0

        # Cycle to the next patient using modulo to wrap around
        self.current_patient_idx = (self.current_patient_idx + 1) % len(
            self.patient_names
        )
        logger.info(
            f"Switching to patient: {self.patient_names[self.current_patient_idx]}"
        )

        # Close the current environment to free resources
        self.env.close()

        # Register and create the new environment for the next patient
        self._register_current_env()
        self.env = gymnasium.make(
            self.current_env_id, render_mode=self._render_mode, seed=self.seed
        )

        # Reset the new environment and return initial observation
        return self.env.reset(**kwargs)


class PatientSwitchCallback(BaseCallback):
    """Custom callback to ensure we cycle through all patients during training.

    This callback works with Stable Baselines3 to:
    1. Force patient switches at regular intervals
    2. Track and log which patient is being used
    3. Ensure the model sees all patients during training

    Without this callback, the model might not cycle through patients
    frequently enough during training.
    """

    def __init__(self, env, switch_freq=1000, verbose=0):
        """Initialize the patient switching callback.

        Args:
            env: The training environment
            switch_freq: How often to force a patient switch (in steps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.env = env
        self.switch_freq = switch_freq
        self.step_count = 0
        self.patient_idx = 0

        # Get the patient names using the proper wrapper access method
        # This is the recommended way to access wrapper attributes in Gymnasium
        if hasattr(env, "get_wrapper_attr"):
            try:
                # Try to get patient_names from wrapped env
                self.patient_names = env.get_wrapper_attr("patient_names")
                self.total_patients = len(self.patient_names)
            except ValueError:
                # If not found in wrappers, try the unwrapped env
                if hasattr(env.unwrapped, "patient_names"):
                    self.patient_names = env.unwrapped.patient_names
                    self.total_patients = len(self.patient_names)
                else:
                    self.total_patients = 1
        else:
            self.total_patients = 1

    def _on_step(self):
        """Called after each step in the environment during training.

        This method:
        1. Tracks training progress
        2. Logs patient changes
        3. Forces patient switching at regular intervals

        Forcing patient switches is crucial because Stable Baselines3's
        internal VecEnv might not respect our environment's episode
        termination signals consistently.

        Returns:
            True to continue training, False to stop
        """
        self.step_count += 1

        # Try to get the current patient index using proper wrapper access
        # This follows Gymnasium's recommended practices for accessing wrapper attributes
        try:
            # First try to get it from wrappers
            if hasattr(self.env, "get_wrapper_attr"):
                current_idx = self.env.get_wrapper_attr("current_patient_idx")
            # Fall back to unwrapped env
            elif hasattr(self.env.unwrapped, "current_patient_idx"):
                current_idx = self.env.unwrapped.current_patient_idx
            else:
                current_idx = 0

            # If the patient has changed, log it
            if self.patient_idx != current_idx:
                self.patient_idx = current_idx
                patient_name = self.patient_names[self.patient_idx]
                logger.info(
                    f"Training on patient {patient_name} ({self.patient_idx + 1}/{self.total_patients})"
                )
        except (ValueError, AttributeError):
            # If we can't get the patient info, just continue
            pass

        # Every switch_freq steps, force a reset to switch patients
        # This ensures we cycle through patients even if episodes are very long
        if self.step_count % self.switch_freq == 0:
            logger.info(
                f"PatientSwitchCallback: Forcing patient switch after {self.switch_freq} steps"
            )
            # Get the VecEnv that SB3 uses internally
            env = self.training_env

            # Force done flag for the current environment
            # This will trigger a reset on the next step
            if hasattr(env, "buf_dones"):
                env.buf_dones[0] = True

        return True  # Continue training


def make_env(cfg, render_mode=None):
    """Create and return a gym environment wrapped with Monitor.

    This function:
    1. Determines which patients to use based on config
    2. Creates a multi-patient environment
    3. Adds monitoring for training metrics

    The environment created here is used for both training and evaluation.

    Args:
        cfg: Configuration dictionary
        render_mode: Visualization mode (if any)

    Returns:
        A properly configured environment for training or evaluation
    """
    # Handle patient selection based on config
    if cfg["env"]["patient_name"] == "all":
        # Use all available default patients
        patient_names = get_default_patients()
    elif isinstance(cfg["env"]["patient_name"], list):
        # Use explicitly listed patients
        patient_names = cfg["env"]["patient_name"]
    else:
        # For single patient, create a list with just that patient
        patient_names = [cfg["env"]["patient_name"]]

    # Create multi-patient environment (works for both single and multiple patients)
    env = MultiPatientEnv(
        patient_names=patient_names,
        env_id=cfg["env"]["id"],
        entry_point=cfg["env"]["entry_point"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        reward_fun=custom_reward_fn,
        seed=cfg["seed"],
        render_mode=render_mode,
    )

    # Add monitoring wrapper for tracking performance
    log_dir = Path(cfg.get("monitor_log_dir", "monitor_logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, filename=str(log_dir))
    return env


def select_device(cfg):
    """Select device based on availability and configuration."""
    # Auto-detect if not provided in the config
    if "device" in cfg and cfg["device"]:
        device = cfg["device"]
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    # Update the config so that downstream functions use the correct device
    cfg["device"] = device
    return device


def train(cfg):
    """Training routine for the DDPG agent."""
    logger.info("Starting training...")
    env = make_env(cfg, render_mode=None)

    n_actions = env.action_space.shape[-1]
    sigma = cfg["action_noise"]["sigma"]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions)
    )

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
        name_prefix="ddpg_checkpoint",
    )

    # Add our custom patient switching callback
    # Switch patients every 500 steps (or adjust as needed)
    patient_switch_freq = min(500, cfg["training"]["total_timesteps"] // 10)
    patient_callback = PatientSwitchCallback(env, switch_freq=patient_switch_freq)

    # Start training
    model.learn(
        total_timesteps=cfg["training"]["total_timesteps"],
        callback=[eval_callback, checkpoint_callback, patient_callback],
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
        model = DDPG.load(
            cfg.get("model_save_path", "ddpg_simglucose"), device=cfg["device"]
        )
    except Exception as e:
        logger.error(f"Error loading model with model_save_path: {e}")
        return

    logger.info(f"Model loaded from '{cfg.get('model_save_path', 'ddpg_simglucose')}'.")
    print(env.reset(seed=cfg["seed"]))
    observation, info = env.reset(seed=cfg["seed"])

    max_steps = cfg.get("predict_steps", 20)
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
    history = env.show_history()
    history.to_csv(f"{cfg['predict']['save_path']}/{cfg['predict']['prefix']}.csv")
    # Close the environment
    env.close()


def analyze(cfg):
    path = Path(__file__).parent
    result_filenames = list(path.glob(f"{cfg['analyze']['files_path']}/*.csv"))
    patient_names = [f.stem for f in result_filenames]
    df = pd.concat(
        [pd.read_csv(str(f), index_col=0) for f in result_filenames], keys=patient_names
    )
    report(df, save_path=cfg["analyze"]["save_path"])


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

    # Setup noise and model
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=cfg["action_noise"]["sigma"] * np.ones(n_actions),
    )

    # Create model with custom network architecture
    model_kwargs = cfg["model"].copy()  # Create a copy of model config
    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        device=cfg["device"],
        policy_kwargs={
            "net_arch": [network_config["hidden_units"]] * network_config["n_layers"]
        },
        **model_kwargs,
    )

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


def main():
    # Parse configuration from YAML
    cfg = parse_args()
    # Set a fixed seed for reproducibility
    np.random.seed(cfg["seed"])

    # Select device and update the configuration
    select_device(cfg)

    # Decide on the mode based on configuration
    mode = cfg.get("mode", "train").lower()

    if mode == "train":
        train(cfg)
    elif mode == "predict":
        predict(cfg)
    elif mode == "grid_search":
        # Run grid search over network configurations
        results = grid_search(cfg)
        # Visualize and save results
        visualize_results(results)
        # Also print results to console
        plot_results(results)
    elif mode == "analyze":
        analyze(cfg)
    else:
        logger.error(
            f"Unknown mode '{mode}'. Please choose 'train', 'predict', 'report', or 'grid_search'."
        )


if __name__ == "__main__":
    main()
