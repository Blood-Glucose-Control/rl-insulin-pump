import gymnasium
from gymnasium.envs.registration import register
import logging
from gymnasium import Wrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        discrete_action_space=False,
        discrete_observation_space=False,
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
        self.discrete_action_space = discrete_action_space
        self.discrete_observation_space = discrete_observation_space

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

        # Parse the base ID into components (handling namespaces and versions correctly)
        # This ensures we create valid Gymnasium IDs like "simglucose/base-adolescent_001-v0"
        if "/" in self.base_env_id:
            # Handle namespace/id format (e.g., "simglucose/something-v0")
            namespace, base_id = self.base_env_id.split("/")
            # Extract the version from the base ID
            if "-v" in base_id:
                base_name, version = base_id.rsplit("-v", 1)
                # Create a new valid ID with namespace
                self.current_env_id = (
                    f"{namespace}/{base_name}-{safe_patient_id}-v{version}"
                )
            else:
                # If no version, just append the patient ID
                self.current_env_id = f"{namespace}/{base_id}-{safe_patient_id}"
        else:
            # Handle IDs without namespace
            if "-v" in self.base_env_id:
                base_name, version = self.base_env_id.rsplit("-v", 1)
                self.current_env_id = f"{base_name}-{safe_patient_id}-v{version}"
            else:
                self.current_env_id = f"{self.base_env_id}-{safe_patient_id}"

        # Set up the environment-specific parameters
        kwargs = {
            "patient_name": patient_name,  # The actual patient identifier
            "reward_fun": self.reward_fun,  # The reward function to use
            "discrete_action_space": self.discrete_action_space,
            "discrete_observation_space": self.discrete_observation_space,
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
    
    def show_history(self):
        self.env.unwrapped.show_history()
