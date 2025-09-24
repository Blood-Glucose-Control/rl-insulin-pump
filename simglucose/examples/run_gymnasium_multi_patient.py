from datetime import datetime
import gymnasium
from gymnasium.envs.registration import register
from simglucose.simulation.scenario import CustomScenario
import numpy as np

start_time = datetime(2018, 1, 1, 0, 0, 0)
meal_scenario_1 = CustomScenario(start_time=start_time, scenario=[(1, 20)])
meal_scenario_2 = CustomScenario(start_time=start_time, scenario=[(3, 15)])


patient_name = [
    "adult#001",
    "adult#002",
    "adult#003",
    "adult#004",
    "adult#005",
    "adult#006",
    "adult#007",
    "adult#008",
    "adult#009",
    "adult#010",
]


register(
    id="simglucose/adolescent2-v0",
    entry_point="simglucose.envs:T1DSimGymnaisumEnv",
    max_episode_steps=10,
    kwargs={
        "patient_name": patient_name,
        "custom_scenario": [meal_scenario_1, meal_scenario_2],
    },
)

env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")
observation, info = env.reset()
for t in range(200):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(
        f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
    )
    if terminated or truncated:
        print("Episode finished after {} timesteps".format(t + 1))
        break
