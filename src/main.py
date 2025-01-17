import gymnasium
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from simglucose.analysis.risk import risk_index # type: ignore

IS_TRAINING = True

# Set a fixed seed
seed = 42
np.random.seed(seed)

# Worth researching if there are more effective reward functions
# maybe consider rate of change of bgl, though risk index may already accounts for this
def custom_reward_fn(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_current - risk_prev

# Register the custom environment
register(
    id="simglucose/adolescent2-v0",
    entry_point="simglucose.envs:T1DSimGymnaisumEnv",
    max_episode_steps=1000,
    kwargs={"patient_name": "adolescent#002", 'reward_fun': custom_reward_fn},
)

def train():
    env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

    model.learn(total_timesteps=10000, log_interval=10)

    # Save trained model
    model.save("ddpg_simglucose")
    print("Model saved as 'ddpg_simglucose'.")

def predict():
    env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human")

    # Load the trained model
    model = DDPG.load("ddpg_simglucose")
    print("Model loaded from 'ddpg_simglucose'.")

    observation, info = env.reset(seed=seed)

    for t in range(200):
        env.render()
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {t}: observation {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}"
        )
        if terminated or truncated:
            print("Episode finished after {} timesteps".format(t + 1))
            break

if __name__ == "__main__":
    train() if IS_TRAINING else predict()