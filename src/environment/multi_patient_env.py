from gymnasium import Wrapper
import gymnasium
import logging
from gymnasium.envs.registration import register
from src.environment.patient import Patient

logger = logging.getLogger(__name__)

class MultiPatientEnv(Wrapper):
    def __init__(
        self,
        patients: list[Patient],
        entry_point: str,
        render_mode: str | None = None,
        seed: int = 0,
        reward_fn = None,
        max_episode_steps: int | None = None,
    ):
        self.patients = patients
        self.entry_point = entry_point
        self._render_mode = render_mode
        self.seed = seed
        self.reward_fn = reward_fn
        self.max_episode_steps = max_episode_steps
        self.current_patient_idx = -1
        self.current_patient = None
        self._register_all_envs()
        env = self._get_next_patient_env()
        super().__init__(env)

    def _register_all_envs(self) -> None:
        """
        Register all patient environments with Gymnasium.
        """
        for patient in self.patients:
            kwargs = {
                "patient_name": patient.patient_id,
                "reward_fn": self.reward_fn,
            }
            try:
                register(
                    id=patient.env_id,
                    entry_point=self.entry_point,
                    max_episode_steps=self.max_episode_steps,
                    kwargs=kwargs,
                )
            except gymnasium.error.Error as e:
                raise Exception(
                    f"Failed to register environment {patient.env_id}: {e}"
                ) from e
            except Exception as e:
                raise Exception(
                    f"An unexpected error occurred while registering {patient.env_id}: {e}"
                ) from e

    def _get_next_patient_env(self) -> gymnasium.Env:
        """
        Get the next patient environment based on the current index.
        This method cycles through the list of patients.
        """
        # Close the previous environment if it exists
        if "env" in self.__dict__ and self.env:
            self.env.close()
        
        # Get the next patient env ID, ensure it wraps around
        self.current_patient_idx = (self.current_patient_idx + 1) % len(self.patients)
        assert self.current_patient_idx >= 0 and self.current_patient_idx < len(
            self.patients
        )
        self.current_patient = self.patients[self.current_patient_idx]

        logger.info(
            f"Switching to patient {self.patients[self.current_patient_idx]} "
        )
        
        # Create gymnasium environment object for the current patient
        return gymnasium.make(
            self.current_patient.env_id, render_mode=self._render_mode, seed=self.seed
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            logger.info(
                f"Episode ended for patient {self.patients[self.current_patient_idx]}."
                f"Terminated: {terminated}, Truncated: {truncated}"
            )
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.env = self._get_next_patient_env()
        return self.env.reset(**kwargs)