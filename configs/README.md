# YAML Configuration Guide

This README is a brief explanation of the configurations available in the yaml file.

## Configurations
| field name    | type  | description|
|---------------|-------|------------|
|device         |STRING |either "cuda", "mps" or "cpu" to indicate which device to use. By default CUDA will be used if available, and CPU will be used otherwise
|mode           |STRING |one of "train", "eval", "predict", or "analyze" to indicate which path to run
|model_save_path|STRING |file to save the model after it is done training, or file to load the model from for evaluation and prediction
|seed           |INT    |the seed to use for random number generation for getting consistent results
|model_name     |STRING |the name of the model to use, currently supports "DDPG" and "PPO"
|run_directory  |STRING |file directory to save all files created by the run
### env
| field name        | type                      | description|
|-------------------|---------------------------|------------|
|id                 |STRING                     |id of the environment to be created
|entry_point        |STRING                     |the Python path of the environment to create, generally "simglucose.envs:T1DSimGymnaisumEnv" for a Simglucose Gymnasium environment, or "simglucose.envs:T1DSimEnv" to not use Gymnasium
|max_episode_steps  |INT                        |the maximum number of steps in an episode
|patient_name       |STRING or (LIST of STRING) |the patient(s) to use in the environment. Use "all" to use all available patients, or a list of patient names or a single patient name to use those patients


### action_noise
| field name    | type  | description|
|---------------|-------|------------|
|mean           |FLOAT  |the mean of the action noise
|sigma          |FLOAT  |the standard deviation of the action noise

### model
Note: since these are used to initialize the model, the model parameters supported can depend on which model is being used (specified by model_name). The below information is for the DDPG model
| field name    | type  | description|
|---------------|-------|------------|
|learning_rate  |FLOAT  |the model's learning rate
|buffer_size    |INT    |the size of the model's buffer
|batch_size     |INT    |the size of the batch
|gamma          |FLOAT  |the model's gamma
|tensorboard_log|PATH   |the directory to store tensorboard logs in

### training
| field name    | type  | description|
|---------------|-------|------------|
|total_timesteps|INT    |the total number of timesteps to train for
|checkpoint_freq|INT    |the number of timesteps between saving checkpoints

### eval
| field name    | type  | description|
|---------------|-------|------------|
|eval_freq      |INT    |number of timesteps between evaluations
|n_eval_episodes|INT    |number of episodes to evaluate the agent for for

### predict
| field name    | type  | description|
|---------------|-------|------------|
|filename       |STRING |the filename of the prediction save file, i.e. file will be <filename\>.csv

### Example Configuration
```yaml
device: cuda
mode: train
model_save_path: trained_model.zip
seed: 42
model_name: DDPG

env:
  id: T1DSimGymnasiumEnv-v0
  entry_point: simglucose.envs:T1DSimGymnaisumEnv
  max_episode_steps: 1000
  patient_name: all

action_noise:
  mean: 0.0
  sigma: 0.1

model:
  learning_rate: 0.001
  buffer_size: 1000000
  batch_size: 64
  gamma: 0.99
  tensorboard_log: logs/

training:
  total_timesteps: 1000000
  checkpoint_freq: 50000

eval:
  eval_freq: 10000
  n_eval_episodes: 10

predict:
  n_eval_episodes: 5
  filename: predictions.csv

```
