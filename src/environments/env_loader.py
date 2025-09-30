from stable_baselines3.common.monitor import Monitor
import logging
from pathlib import Path

from src.environments.multipatient import MultiPatientEnv
from src.environments.reward_functions import risk_diff_reward_fn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_default_patients():
    """Return a list of default patient names that are referenced from [simglucose](https://github.com/Blood-Glucose-Control/rl-insulin-pump/blob/main/simglucose/simglucose/params/vpatient_params.csv)."""
    patients = []

    # Add adolescent patients (typically 001-010)
    for i in range(1, 11):
        patients.append(f"adolescent#{i:03d}")

    # Add adult patients (typically 001-010)
    for i in range(1, 11):
        patients.append(f"adult#{i:03d}")

    # Add child patients (typically 001-010)
    for i in range(1, 11):
        patients.append(f"child#{i:03d}")

    logger.info(f"Using {len(patients)} default patients")
    return patients


def make_env(cfg, mode="train", render_mode=None):
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
        logger.info(f"Using all default patients for {mode}ing.")
    elif isinstance(cfg["env"]["patient_name"], list):
        # Use explicitly listed patients
        patient_names = cfg["env"]["patient_name"]
    else:
        # For single patient, create a list with just that patient
        patient_names = [cfg["env"]["patient_name"]]

    # Create multi-patient environment (works for both single and multiple patients)
    logger.info(f"Creating environment with patients: {patient_names}")

    env = MultiPatientEnv(
        patient_names=patient_names,
        env_id=cfg["env"]["id"],
        entry_point=cfg["env"]["entry_point"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        reward_fun=risk_diff_reward_fn,  # TODO: make this configurable
        seed=cfg["seed"],
        render_mode=render_mode,
        discrete_action_space=cfg["env"].get("discrete_action_space", False),
        discrete_observation_space=cfg["env"].get("discrete_observation_space", False),
        discrete_action_step=cfg["env"].get("discrete_action_step", 0.1),
        discrete_observation_step=cfg["env"].get("discrete_observation_step", 0.1),
    )

    # Add monitoring wrapper for tracking performance
    log_dir = Path(cfg.get("monitor_log_dir", "monitor_logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, filename=str(log_dir))
    return env
