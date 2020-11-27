from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import gym.spaces
import numpy as np
import random

#import gym_compete.envs
#from gym_compete.wrappers.pong_wrappers import wrap_pong

class WrapCompete(gym.Wrapper):
    def __init__(self, env):
        """ Wrap compete envs
        """
        gym.Wrapper.__init__(self, env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return np.array(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(obs, tuple):
            obs = np.array(obs)
        if isinstance(reward, tuple):
            reward = np.array(reward)

        return obs, reward, done, info


class TransposeWrapper(gym.ObservationWrapper):
  def observation(self, observation):
    if isinstance(observation, tuple):
      return tuple([
        np.transpose(np.array(ob), axes=(2,0,1))
        for ob in observation
      ])
    else:
      return np.transpose(np.array(observation), axes=(2,0,1))

class NoRwdResetEnv(gym.Wrapper):
  def __init__(self, env, no_reward_thres):
    """Reset the environment if no reward received in N steps
    """
    gym.Wrapper.__init__(self, env)
    self.no_reward_thres = no_reward_thres
    self.no_reward_step = 0

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if isinstance(reward, tuple):
      if all(r == 0.0 for r in reward):
        self.no_reward_step += 1
      else:
        self.no_reward_step = 0
    else:
      if reward == 0.0:
        self.no_reward_step += 1
      else:
        self.no_reward_step = 0

    if self.no_reward_step > self.no_reward_thres:
      done = True
    return obs, reward, done, info

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self.no_reward_step = 0
    return obs

#def make_pong(env_id, episode_life=True, clip_rewards=True, frame_stack=True, scale=True, seed=None):
#  env = wrap_pong(env_id, episode_life, clip_rewards, frame_stack, scale, seed)
#  #env = TransposeWrapper(env)
#  env = NoRwdResetEnv(env, no_reward_thres = 1000)
#  env = WrapCompete(env)
#  return env
