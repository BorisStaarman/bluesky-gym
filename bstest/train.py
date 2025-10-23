import bluesky_gym
import gymnasium as gym
from stable_baselines3 import PPO, SAC
import bluesky_gym.envs
from bluesky_gym.utils import logger
bluesky_gym.register_envs()

env_name = 'SectorCREnv-v0'

env = gym.make(env_name, render_mode='human')
file_name = 'first_test_model_boris.csv'
logger = logger.CSVLoggerCallback('logs/', file_name)

# train model
model = SAC('MultiInputPolicy', env=env, verbose=1)
model.learn(total_timesteps=100, callback=logger)
model.save("models/first_test_model_boris")
env.close()

model = SAC.load("models/first_test_model_boris")
# env = gym.make(env_name, render_mode='human')

n_eps = 10
for i in range(n_eps):
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()