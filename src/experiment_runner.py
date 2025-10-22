import logging
from simglucose.simulation.env import Observation
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from src.training.callbacks.patient_switch import PatientSwitchCallback
from src.environments.env_loader import make_env
from src.agents.agent_loader import make_model, load_model
from src.utils.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, cfg: Config, callbacks=None):
        self.cfg = cfg
        self.env = make_env(cfg, render_mode=None)
        self.eval_env = make_env(cfg, render_mode=None)
        self.model = make_model(cfg, self.env)

        self.callbacks = [callbacks] if callbacks else []
        if self.cfg.model_name not in ["PID", "BB"]:
            # Evaluation callback
            self.callbacks.append(
                EvalCallback(
                    self.eval_env,
                    best_model_save_path=self.cfg.best_model_path,
                    log_path=self.cfg.eval_log_path,
                    eval_freq=self.cfg.eval.eval_freq,
                    n_eval_episodes=self.cfg.eval.n_eval_episodes,
                )
            )

            # Checkpoint callback
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.cfg.training.checkpoint_freq,
                    save_path=self.cfg.checkpoint_path,
                    name_prefix=f"{self.cfg.model_name}_checkpoint",
                )
            )

            # Patient switch callback
            switch_freq = min(500, self.cfg.training.total_timesteps // 10)
            self.callbacks.append(
                PatientSwitchCallback(self.env, switch_freq=switch_freq)
            )

    def train(self):
        model_name = self.cfg.model_name
        if model_name in ["PID", "BB"]:
            logger.error(f"Cannot run `train` on {model_name} controller")
            return AttributeError(
                f"{model_name} Controller does not have `learn` method"
            )

        logger.info("Starting training...")
        self.model.learn(
            total_timesteps=self.cfg.training.total_timesteps,
            callback=self.callbacks,
        )
        model_path = self.cfg.model_save_path
        self.model.save(model_path)
        logger.info(f"Model saved as '{model_path}'.")

    def predict(self):
        logger.info("Starting prediction...")
        env = make_env(self.cfg, render_mode="human")

        try:
            model = load_model(self.cfg)
        except Exception as e:
            logger.error(f"Error loading model with model_save_path: {e}")
            return

        logger.info(f"Model loaded from '{self.cfg.model_save_path}'.")

        max_steps = self.cfg.predict.predict_steps
        logger.info(f"Running prediction for {max_steps} steps...")

        for patient in env.patient_names:
            observation, info = env.reset(seed=self.cfg.seed)
            logger.info(f"Starting prediction for patient {patient}...")
            for t in range(max_steps):
                env.render()
                if self.cfg.model_name in ["PID", "BB"]:
                    obs = Observation(
                        CGM=observation[0]
                    )  # Simglucose controllers (PID and BBC) expect observation to be an Observation instance
                    action = self.model.policy(obs, 0, False, **info)
                    action = action.basal + action.bolus
                else:
                    action, _ = model.predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                logger.info(
                    f"Step {t}: obs {observation}, reward {reward}, term {terminated}, trunc {truncated}, info {info}"
                )
                if terminated or truncated:
                    logger.info(
                        "Episode for patient {} finished after {} timesteps".format(
                            patient, t + 1
                        )
                    )
                    break
            history = env.unwrapped.show_history()
            history.to_csv(
                f"{self.cfg.run_directory}/results/predict/{patient}_predict.csv"
            )

        env.close()
