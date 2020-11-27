from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from random import choice

import gym
from gym.spaces import Box, Discrete


class EpisodicLifeEnv(gym.Wrapper):
  """Make end-of-life == end-of-episode, but only reset on true game over.
  Done by DeepMind for the DQN and co. since it helps value estimation.
  """

  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self.was_real_done = True

  def reset(self, **kwargs):
    """Reset only when lives are exhausted.
    This way all states are still reachable even though lives are episodic,
    and the learner need not know about any of this behind-the-scenes.
    """
    if self.was_real_done:
      obs = self.env.reset(**kwargs)
    else:
      obs, _, _, _ = self.env.step(self.env.unwrapped.action_use)
    return obs

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.was_real_done = done
    done = self.env.unwrapped.game.is_player_dead() or self.was_real_done
    return obs, reward, done, info


class SkipFrameVEnv(gym.Wrapper):
  """Return only every `skip`-th frame for vec env

  Note: in multi-player game via network connection, the most safe way seems to
  ensure all players simultaneously make ONLY ONE step once a time. See:
  https://github.com/mwydmuch/ViZDoom/issues/261
  """

  def __init__(self, env, skip=4):
    gym.Wrapper.__init__(self, env)
    self._skip = skip

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    """Repeat action, sum reward"""
    total_reward = None
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if total_reward is None:
        total_reward = reward
      else:
        total_reward = [a+b for a, b in zip(total_reward, reward)]
      if all(done):
        break
    return obs, total_reward, done, info


class SkipFrameEnv(gym.Wrapper):
  """Return only every `skip`-th frame for PlayerEnv

  Seems problematic in multiplayer mode, see
  https://github.com/mwydmuch/ViZDoom/issues/261
  """

  def __init__(self, env, skip=4):
    gym.Wrapper.__init__(self, env)
    self._skip = skip

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    """Repeat action, sum reward"""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      total_reward += reward
      if done:
        break
    return obs, total_reward, done, info


class RepeatFrameEnv(gym.Wrapper):
  """Repeat Frame n times for PlayerEnv

  Seems problematic in multiplayer mode, see
  https://github.com/mwydmuch/ViZDoom/issues/261"""

  def __init__(self, env, n=4):
    gym.Wrapper.__init__(self, env)
    self._n = n
    self.env.unwrapped.cfg.repeat_frame = self._n

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

  def step(self, action):
    return self.env.step(action)


class RandomConfigVEnv(gym.Wrapper):
  """ Start Doom game with randomly selected cfg file, for vec env"""

  def __init__(self, env, cfg_path_list):
    gym.Wrapper.__init__(self, env)
    self._cfg_path_list = cfg_path_list

  def reset(self, **kwargs):
    cfg_path = choice(self._cfg_path_list)
    for e in self.env.unwrapped.envs:
      e.unwrapped.cfg.config_path = cfg_path
    return self.env.reset(**kwargs)

  def step(self, action):
    return self.env.step(action)
