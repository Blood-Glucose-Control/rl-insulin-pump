import gymnasium
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from simglucose.analysis.risk import risk_index # type: ignore
import torch
from cmd_args import parse_args

args_dict = parse_args()
IS_TRAINING = True


# Set a fixed seed
seed = args_dict['seed']
np.random.seed(seed)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

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
    id=args_dict['env']['id'],
    entry_point=args_dict['env']['entry_point'],
    max_episode_steps=args_dict['env']['max_episode_steps'],
    kwargs={"patient_name": args_dict['env']['patient_name'], 'reward_fun': custom_reward_fn},
)

def train():
    
    env = gymnasium.make("simglucose/adolescent2-v0", render_mode=None, seed=seed)
    env = Monitor(env, filename="monitor_logs/")
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, device = device,
                 learning_rate=1e-3, buffer_size=100000, batch_size=256, gamma=0.99, tensorboard_log="./tb_logs/")
    
    eval_env = gymnasium.make("simglucose/adolescent2-v0", render_mode=None, seed=seed)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", 
                                 log_path="./logs/", eval_freq=5000, n_eval_episodes=5)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/', name_prefix='ddpg_checkpoint')
    model.learn(total_timesteps=200000, callback=[eval_callback, checkpoint_callback])

    # Save trained model
    model.save("ddpg_simglucose")
    print("Model saved as 'ddpg_simglucose'.")

def predict():
    env = gymnasium.make("simglucose/adolescent2-v0", render_mode="human", seed=seed)
    env = Monitor(env, filename="monitor_logs/")
    # Load the trained model
    model = DDPG.load("ddpg_simglucose", device=device)
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