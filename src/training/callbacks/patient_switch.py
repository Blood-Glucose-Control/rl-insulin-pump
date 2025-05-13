import logging
from stable_baselines3.common.callbacks import (
    BaseCallback,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
