"""
Minimal configuration class for RL Insulin Pump.

This module provides a simple wrapper around the existing dictionary-based
config system to provide a cleaner interface without breaking changes.
"""

from typing import Any, Dict, Union
from pathlib import Path
import yaml
import logging
import torch

logger = logging.getLogger(__name__)


class EnvConfig:
    """Environment configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "simglucose/multi-patient-v0")
        self.entry_point = data.get("entry_point", "simglucose.envs:T1DSimGymnaisumEnv")
        self.max_episode_steps = data.get("max_episode_steps", 1000)
        self.discrete_action_space = data.get("discrete_action_space", False)
        self.discrete_observation_space = data.get("discrete_observation_space", False)

        # Normalize patient_name to always be a list
        patient_names = data.get("patient_name", "all")
        if patient_names == "all":
            from src.environments.env_loader import get_default_patients
            self.patient_names = get_default_patients()
        elif isinstance(patient_names, list):
            self.patient_names = patient_names
        else:
            self.patient_names = [patient_names]


class ModelConfig:
    """Model configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        self.policy = data.get("policy", "MlpPolicy")
        self.learning_rate = data.get("learning_rate", 0.001)
        self.buffer_size = data.get("buffer_size", 100000)
        self.batch_size = data.get("batch_size", 256)
        self.gamma = data.get("gamma", 0.99)


class TrainingConfig:
    """Training configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        self.total_timesteps = data.get("total_timesteps", 1000)
        self.checkpoint_freq = data.get("checkpoint_freq", 10000)

class EvalConfig:
    """Evaluation configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        self.eval_freq = data.get("eval_freq", 5000)
        self.n_eval_episodes = data.get("n_eval_episodes", 5)


class PredictConfig:
    """Prediction configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        self.prefix = data.get("prefix", "prediction")
        self.predict_steps = data.get("predict_steps", 40)
        self.filename = data.get("filename", "glucose_history")
        if not self.filename.endswith(".csv"):
            self.filename = f"{self.filename}.csv"


class ActionNoiseConfig:
    """Action noise configuration namespace."""

    def __init__(self, data: Dict[str, Any]):
        self.mean = data.get("mean", 0.0)
        self.sigma = data.get("sigma", 0.25)


class Config:
    """Minimal configuration class that wraps the existing dictionary structure."""

    def __init__(self, data: Dict[str, Any]):
        """Initialize config with dictionary data."""
        self._data = data

        # Set common attributes directly for easy access
        self.seed = data.get("seed", 42)
        self.device = data.get("device", "cpu")
        self.model_name = data.get("model_name", "DDPG")
        self.run_directory = data.get("run_directory", "RUN_DIR")
        self.modes = data.get("modes", ["train"])
        # Default model_save_path based on model_name
        self.model_save_path = data.get(
            "model_save_path", f"{self.model_name.lower()}_simglucose"
        )
        self.monitor_log_dir = data.get("monitor_log_dir", "monitor_logs/")

        # Validate device on initialization
        self._validate_device()

        # Create namespace objects for nested configs
        self.env = EnvConfig(data.get("env", {}))
        self.model = ModelConfig(data.get("model", {}))
        self.training = TrainingConfig(data.get("training", {}))
        self.eval = EvalConfig(data.get("eval", {}))
        self.predict = PredictConfig(data.get("predict", {}))
        self.action_noise = ActionNoiseConfig(data.get("action_noise", {}))

        # Set path attributes directly
        self.tensorboard_log_path = f"{self.run_directory}/tensorboard/"
        self.best_model_path = f"{self.run_directory}/best_model/"
        self.checkpoint_path = f"{self.run_directory}/checkpoints/"
        self.eval_log_path = f"{self.run_directory}/logs/"
        self.predict_results_path = f"{self.run_directory}/results/predict/"
        self.analysis_results_path = f"{self.run_directory}/results/analysis/"

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file {yaml_path} not found")

        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default, maintaining backward compatibility."""
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for backward compatibility."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment for backward compatibility."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._data

    def __str__(self) -> str:
        """String representation of config."""
        return f"Config(model_name={self.model_name}, device={self.device}, mode={self.mode})"

    def __repr__(self) -> str:
        """Detailed representation of config."""
        return f"Config(seed={self.seed}, device={self.device}, model_name={self.model_name}, run_directory={self.run_directory})"

    def _validate_device(self):
        """Validate the device configuration."""
        if self.device not in ["cuda", "mps", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        logger.info(f"Device validation successful: {self.device}")
