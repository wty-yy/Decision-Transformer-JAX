import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np

env = 'halfcheetah-medium-expert-v2'

# Create the environment
env = gym.make(env)

# d4rl abides by the OpenAI gym interface
s = env.reset()
s, r, t, info = env.step(env.action_space.sample())
print(s, r, t, info)

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations
print(dataset.keys())
print(dataset['observations'].shape)
print(dataset['actions'].shape)
obs, a, r, t = dataset['observations'], dataset['actions'], dataset['rewards'], dataset['timeouts']
print(a.min(), a.max())
print(r.min(), r.max())
print(t)
print(t)
print(np.where(t))
print(a.dtype)

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
# dataset = d4rl.qlearning_dataset(env)