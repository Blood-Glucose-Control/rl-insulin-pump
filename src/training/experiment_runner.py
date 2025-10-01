import logging
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from src.training.callbacks.patient_switch import PatientSwitchCallback
from src.environments.env_loader import make_env
from src.agents.agent_loader import make_model, load_model
from src.utils.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, cfg, config: Config, callbacks=None):
        self.cfg = cfg
        self.config = config
        self.env = make_env(config, render_mode=None)
        self.eval_env = make_env(config, render_mode=None)
        self.model = make_model(config, self.env)

        self.callbacks = [callbacks] if callbacks else []
        # Evaluation callback
        self.callbacks.append(
            EvalCallback(
                self.eval_env,
                best_model_save_path=self.config.best_model_path,
                log_path=self.config.eval_log_path,
                eval_freq=self.config.eval.eval_freq,
                n_eval_episodes=self.config.eval.n_eval_episodes,
            )
        )

        # Checkpoint callback
        self.callbacks.append(
            CheckpointCallback(
                save_freq=self.config.training.checkpoint_freq,
                save_path=self.config.checkpoint_path,
                name_prefix=f"{self.config.model_name}_checkpoint",
            )
        )

        # Patient switch callback
        switch_freq = min(500, self.config.training.total_timesteps // 10)
        self.callbacks.append(PatientSwitchCallback(self.env, switch_freq=switch_freq))

    def train(self):
        logger.info("Starting training...")
        self.model.learn(
            total_timesteps=self.config.training.total_timesteps,
            callback=self.callbacks,
        )
        model_path = self.config.model_save_path
        self.model.save(model_path)
        logger.info(f"Model saved as '{model_path}'.")

    def multi_patient_predict(self, env, model, max_steps=40):
        for patient in env.patient_names:
            observation, info = env.reset(seed=self.cfg["seed"])
            logger.info(f"Starting prediction for patient {patient}...")
            for t in range(max_steps):
                env.render()
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
                f"{self.cfg['run_directory']}/results/predict/{patient}_predict.csv"
            )

    def predict(self):
        logger.info("Starting prediction...")
        env = make_env(self.config, render_mode="human")

        try:
            model = load_model(self.config)
        except Exception as e:
            logger.error(f"Error loading model with model_save_path: {e}")
            return

        logger.info(f"Model loaded from '{self.config.model_save_path}'.")

        predict_cfgs = self.cfg.get("predict", {})
        if "predict_steps" in predict_cfgs:
            max_steps = predict_cfgs["predict_steps"]
        else:
            max_steps = 40
            logger.info(f"Running prediction for default {max_steps} steps...")
        logger.info(f"Running prediction for {max_steps} steps...")

        if self.cfg["env"]["patient_name"] == "all" or isinstance(
            self.cfg["env"]["patient_name"], list
        ):
            self.multi_patient_predict(env, model, max_steps)
        else:
            observation, info = env.reset(seed=self.cfg["seed"])
            for t in range(max_steps):
                env.render()
                action, _ = model.predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                logger.info(
                    f"Step {t}: obs {observation}, reward {reward}, term {terminated}, trunc {truncated}, info {info}"
                )
                if terminated or truncated:
                    logger.info("Episode finished after {} timesteps".format(t + 1))
                    break

            history = env.unwrapped.show_history()
            history.to_csv(
                f"{self.config.predict_results_path}{self.config.predict.filename}"
            )
        env.close()
