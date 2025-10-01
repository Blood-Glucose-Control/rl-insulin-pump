from pathlib import Path
import yaml

from src.utils.config import Config


SAMPLE_YAML = """
env:
  id: "simglucose/multi-patient-v0"
  entry_point: "simglucose.envs:T1DSimGymnaisumEnv"
  max_episode_steps: 10000
  patient_name: "all" 
seed: 42
run_directory: "RUN_DIR"
model_name: "DDPG"
action_noise:
  mean: 0
  sigma: 0.25
model:
  learning_rate: 0.001
  buffer_size: 100000
  batch_size: 256
  gamma: 0.99
  policy: "MlpPolicy"
training:
  total_timesteps: 10000
  checkpoint_freq: 1000
eval:
  eval_freq: 5000
  n_eval_episodes: 5
predict:
  prefix: "adolescent#002"
  predict_steps: 20
device: "cpu"
modes: ["predict"]
"""


def write_temp_yaml(tmp_path: Path) -> Path:
    """
    Write a sample yaml file to a temporary directory.
    """
    cfg_path = tmp_path / "sample_config.yaml"
    cfg_path.write_text(SAMPLE_YAML)
    return cfg_path


def test_config_parsing_matches_yaml(tmp_path):
    """
    Test that Config parsing matches raw YAML parsing.
    tmp_path is a pytest fixture that provides a unique temporary directory.
    This pattern ensures each test gets its own isolated temporary space without conflicts.
    After the test completes, pytest automatically cleans up the temporary directory.
    """
    # Arrange: write sample yaml
    yaml_path = write_temp_yaml(tmp_path)

    # Act: parse with yaml and with our Config
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = Config.from_yaml(yaml_path)

    # Assert top-level fields
    assert cfg.seed == raw["seed"]
    assert cfg.device == raw["device"]
    assert cfg.model_name == raw["model_name"]
    assert cfg.run_directory == raw["run_directory"]
    assert cfg.modes == raw["modes"]

    # Assert env namespace
    assert cfg.env.id == raw["env"]["id"]
    assert cfg.env.entry_point == raw["env"]["entry_point"]
    assert cfg.env.max_episode_steps == raw["env"]["max_episode_steps"]
    assert cfg.env.patient_name == raw["env"]["patient_name"]

    # Assert model namespace
    assert cfg.model.learning_rate == raw["model"]["learning_rate"]
    assert cfg.model.buffer_size == raw["model"]["buffer_size"]
    assert cfg.model.batch_size == raw["model"]["batch_size"]
    assert cfg.model.gamma == raw["model"]["gamma"]
    assert cfg.model.policy == raw["model"]["policy"]

    # Assert training namespace
    assert cfg.training.total_timesteps == raw["training"]["total_timesteps"]
    assert cfg.training.checkpoint_freq == raw["training"]["checkpoint_freq"]

    # Assert eval namespace
    assert cfg.eval.eval_freq == raw["eval"]["eval_freq"]
    assert cfg.eval.n_eval_episodes == raw["eval"]["n_eval_episodes"]

    # Assert predict namespace
    assert cfg.predict.prefix == raw["predict"]["prefix"]
    assert cfg.predict.predict_steps == raw["predict"]["predict_steps"]

    # Assert action noise namespace
    assert cfg.action_noise.mean == raw["action_noise"]["mean"]
    assert cfg.action_noise.sigma == raw["action_noise"]["sigma"]

    # Assert derived paths
    assert cfg.tensorboard_log_path.endswith("/tensorboard/")
    assert cfg.best_model_path.endswith("/best_model/")
    assert cfg.checkpoint_path.endswith("/checkpoints/")
    assert cfg.eval_log_path.endswith("/logs/")
    assert cfg.predict_results_path.endswith("/results/predict/")
    assert cfg.analysis_results_path.endswith("/results/analysis/")
