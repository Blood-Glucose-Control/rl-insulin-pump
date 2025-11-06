from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import numpy as np
import pkg_resources
import gym
from gym import spaces
from gym.utils import seeding
from datetime import datetime
import gymnasium


PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)

MAX_TIME_SINCE_MEAL = 24 * 60  # 24h


class T1DSimEnv(gym.Env):
    """
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    """

    metadata = {"render.modes": ["human"]}

    SENSOR_HARDWARE = "Dexcom"
    INSULIN_PUMP_HARDWARE = "Insulet"

    def __init__(
        self, patient_name=None, custom_scenario=None, reward_fun=None, seed=None, include_carb_announcement=False
    ):
        """
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        """
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            patient_name = ["adolescent#001"]

        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.custom_scenario = custom_scenario
        self.include_carb_announcement = include_carb_announcement
        self.last_meal_time = None
        self.env, _, _, _ = self._create_env()

    def _step(self, action: float):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def _raw_reset(self):
        return self.env.reset()

    def _reset(self):
        self.env, _, _, _ = self._create_env()
        obs, _, _, _ = self.env.reset()
        self.last_meal_time = None
        return obs

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env()
        return [seed1, seed2, seed3, seed4]

    def _create_env(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2**31
        seed3 = seeding.hash_seed(seed2 + 1) % 2**31
        seed4 = seeding.hash_seed(seed3 + 1) % 2**31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)

        if isinstance(self.patient_name, list):
            patient_name = self.np_random.choice(self.patient_name)
            patient = T1DPatient.withName(patient_name, random_init_bg=True, seed=seed4)
        else:
            patient = T1DPatient.withName(
                self.patient_name, random_init_bg=True, seed=seed4
            )

        if isinstance(self.custom_scenario, list):
            scenario = self.np_random.choice(self.custom_scenario)
        else:
            scenario = (
                RandomScenario(start_time=start_time, seed=seed3)
                if self.custom_scenario is None
                else self.custom_scenario
            )

        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def _render(self, mode="human", close=False):
        self.env.render(close=close)

    def _close(self):
        super()._close()
        self.env._close_viewer()

    def show_history(self):
        return self.env.show_history()
    
    @property
    def action_space(self):
        ub = self.env.pump._params["max_basal"]
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        if self.include_carb_announcement:
            # [glucose, carb_announcement, time_since_meal]
            return spaces.Box(low=0, high=1000, shape=(3,), dtype=np.float32)
        else:
            return spaces.Box(low=0, high=1000, shape=(1,))

    @property
    def max_basal(self):
        return self.env.pump._params["max_basal"]

    def _get_carbs_at_time(self):
        """Get carbohydrate amount from current scenario step"""
        if not hasattr(self.env, 'scenario') or not hasattr(self.env.scenario, 'scenario'):
            return 0.0
        
        # Get current time from the environment (in minutes)
        current_time = self.env.time
        start_time = self.env.scenario.start_time
        time_diff = (current_time - start_time).total_seconds() / 60.0
        
        scenario = self.env.scenario.scenario
        if 'meal' not in scenario:
            return 0.0
        
        
        meal_times = scenario['meal']['time']
        meal_amounts = scenario['meal']['amount']
        
        # Check if current time matches any meal time (within tolerance)
        for meal_time, carb_amount in zip(meal_times, meal_amounts):
            # meal_time is in minutes
            if abs(time_diff - meal_time) < 3:  # within 1 step (tolerance)
                return float(carb_amount)
        return 0.0

    def _get_time_since_meal(self):
        """Get minutes since last meal announcement"""
        if self.last_meal_time is None:
            return MAX_TIME_SINCE_MEAL
        
        # Both are datetime objects, so subtraction gives timedelta
        time_diff = self.env.time - self.last_meal_time
        time_since_minutes = time_diff.total_seconds() / 60.0
        return min(time_since_minutes, 1440.0)

    def _build_observation(self, obs):
        """Build extended observation with carb info if enabled"""
        if self.include_carb_announcement:
            carb = self._get_carbs_at_time()
            
            # Update meal tracking
            if carb > 0:
                self.last_meal_time = self.env.time
            
            time_since_meal = self._get_time_since_meal()
            
            return np.array([obs.CGM, carb, time_since_meal], dtype=np.float32)
        else:
            return np.array([obs.CGM], dtype=np.float32)


class T1DSimGymnaisumEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    MAX_BG = 1000

    def __init__(
        self,
        patient_name=None,
        custom_scenario=None,
        reward_fun=None,
        seed=None,
        render_mode=None,
        discrete_action_space=False,
        discrete_observation_space=False,
        include_carb_announcement=False,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.env = T1DSimEnv(
            patient_name=patient_name,
            custom_scenario=custom_scenario,
            reward_fun=reward_fun,
            seed=seed,
            include_carb_announcement=include_carb_announcement,
        )
        self.discrete_action_space = discrete_action_space
        self.discrete_observation_space = discrete_observation_space
        self.include_carb_announcement = include_carb_announcement
        
        # for dqn models, we want a discrete action space and observation space instead
        if discrete_action_space:
            self.action_space = gymnasium.spaces.Discrete(int((self.env.max_basal)//0.1), start=0)
        else:
            self.action_space = gymnasium.spaces.Box(
                low=0, high=self.env.max_basal, shape=(1,), dtype=np.float32
            )
        
        if discrete_observation_space:
            self.observation_space = gymnasium.spaces.Discrete(int((self.MAX_BG)//0.1), start=0)
        else:
            if self.include_carb_announcement:
                # [glucose, carb_announcement, time_since_meal]
                self.observation_space = gymnasium.spaces.Box(
                    low=0, high=1000, shape=(3,), dtype=np.float32
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    low=0, high=self.MAX_BG, shape=(1,), dtype=np.float32
                )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        
        if self.discrete_observation_space:
            return int(obs.CGM*10), reward, done, truncated, info
        
        # Build observation with carbs if enabled
        obs_array = self.env._build_observation(obs)
        return obs_array, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, _, _, info = self.env._raw_reset()
        self.env.last_meal_time = None
        
        if self.discrete_observation_space:
            return int(obs.CGM*10), info
        
        obs_array = self.env._build_observation(obs)
        return obs_array, info

    def render(self):
        if self.render_mode == "human":
            self.env.render()

    def close(self):
        self.env.close()

    def show_history(self):
        return self.env.unwrapped.show_history()