from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from copy import deepcopy

import gym
from gym.spaces import Box, Discrete
import numpy as np
import cv2
import vizdoom as vd


class PermuteAndResizeFrame(gym.ObservationWrapper):
  """ CHW to HWC, and Resize Frame """
  def __init__(self, env, height=84, width=84):
    gym.ObservationWrapper.__init__(self, env)
    channel = self.env.observation_space.shape[0]
    self.observation_space = self.env.observation_space
    self.observation_space.shape = (height, width, channel)

    self._height, self._width = height, width

  def observation(self, frame):
    frame = np.transpose(frame, axes=(1, 2, 0))
    frame = cv2.resize(frame, (self._height, self._width),
                       interpolation=cv2.INTER_AREA)
    return frame


class PermuteFrame(gym.ObservationWrapper):
  """ CHW to HWC, and Resize Frame """
  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    c, h, w = self.env.observation_space.shape
    self.observation_space = self.env.observation_space
    self.observation_space.shape = (h, w, c)

  def observation(self, frame):
    frame = np.transpose(frame, axes=(1, 2, 0))
    return frame


class WuObservation(gym.ObservationWrapper):
  """ Wu Yuxin's Observation. Screen + GameVariables.

  Expose observations as a list [screen, game_var], where
  screen.shape = (height, width, channel)
  and
  game_var.shape = (2,)
  which includes (health, ammo) normalized to [0.0, 1.0]
  """
  def __init__(self, env, height=84, width=84):
    gym.ObservationWrapper.__init__(self, env)
    # export observation space
    channel = self.env.observation_space.shape[0]
    screen_sp = self.env.observation_space
    screen_sp.shape = (height, width, channel)
    game_var_sp = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    self.observation_space = [screen_sp, game_var_sp]

    self._height, self._width = height, width
    self._dft_gamevar = np.zeros(shape=game_var_sp.shape,
                                 dtype=game_var_sp.dtype)
    self._gamevar = deepcopy(self._dft_gamevar)

  def observation(self, frame):
    # Permute and resize
    frame = np.transpose(frame, axes=(1, 2, 0))
    frame = cv2.resize(frame, (self._height, self._width),
                       interpolation=cv2.INTER_AREA)
    # normalized game vars
    self._grab_gamevar()
    return [frame, self._gamevar]

  def _grab_gamevar(self):
    if self.env.unwrapped._state is not None:
      game = self.env.unwrapped.game
      self._gamevar[0] = game.get_game_variable(vd.GameVariable.HEALTH) / 100.0
      self._gamevar[1] = game.get_game_variable(
        vd.GameVariable.SELECTED_WEAPON_AMMO
      ) / 15.0
