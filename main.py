"""
This file is an example train and test loop for the different environments.
Selecting different environments is done through setting the 'env_name' variable.

TODO:
* add rgb_array rendering for the different environments to allow saving videos
"""

import gymnasium as gym
from stable_baselines3 import PPO, SAC, TD3, DDPG

import numpy as np

import bluesky_gym
import bluesky_gym.envs

from bluesky_gym.utils import logger # custom utility logger from bluesky

bluesky_gym.register_envs() # register the environments

#env_name = 'HorizontalCREnv-v0' # specify environment
env_name = 'SectorCREnv-v0_boris' # specify environment you want to train or test
algorithm = SAC # choose one of PPO, SAC, TD3, DDPG RL algorithms

# Initialize logger
log_dir = f'./logs/{env_name}/' # log directory
file_name = f'{env_name}_{str(algorithm.__name__)}.csv' # file name for the csv logger
csv_logger_callback = logger.CSVLoggerCallback(log_dir, file_name) 

TRAIN = False # True to train, False to only evaluate
EVAL_EPISODES = 10 # number of episodes to evaluate the trained model, after training


if __name__ == "__main__":
    env = gym.make(env_name, render_mode=None) # create the environment
    obs, info = env.reset() # reset to starting state
    # initialize the model, specifying data that needs to be known, verbose = 1 means we can see the training output
    # calls to the init function in the algorithm.py file
    model = algorithm("MultiInputPolicy", env, verbose=1,learning_rate=3e-4) 
    if TRAIN: # if we want to train the model, this reffers to the functions learn and save in the algorithm.py file
        model.learn(total_timesteps=2e4, callback=csv_logger_callback) # train the model, change total_timesteps, original 2e6,
        model.save(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model")
        del model
    env.close()
    
    # Test the trained model
    model = algorithm.load(f"models/{env_name}/{env_name}_{str(algorithm.__name__)}/model", env=env) # load already trained or just trained model 
    env = gym.make(env_name, render_mode="human") # create the environment with rendering
    for i in range(EVAL_EPISODES): # number of episodes to play
        # initialize states for each episode
        done = truncated = False
        obs, info = env.reset()
        tot_rew = 0
        while not (done or truncated):
            # action = np.array(np.random.randint(-100,100,size=(2))/100)
            # action = np.array([0,-1])
            action, _states = model.predict(obs, deterministic=True) # choose best action according to the trained model
            obs, reward, done, truncated, info = env.step(action[()]) # take action and return new state, reward, done, truncated, info
            tot_rew += reward
        print('reward:',tot_rew)
    env.close()