from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from math import sqrt
from copy import deepcopy

import gym
from gym import spaces
import numpy as np
import vizdoom as vd


def calc_dist(pos, pos_prev):
  x, y = pos
  xx, yy = pos_prev
  dx, dy = x - xx, y - yy
  return sqrt(dx * dx + dy * dy)


class RwdShapeWuBasic(gym.RewardWrapper):
  """ Wu Yuxing's reward shaping.  """
  IND_GAMEVAR_FRAG = 0
  IND_GAMEVAR_HEALTH = 1
  IND_GAMEVAR_AMMO = 2
  EPISODE_REAL_START = 1
  LIFE_REAL_START = 3

  def __init__(self, env, is_recompute_reward=True, dist_penalty_thres=1):
    super(RwdShapeWuBasic, self).__init__(env)
    self.is_recompute_reward = is_recompute_reward
    self.dist_penalty_thres = dist_penalty_thres

    self.game_variables = None
    self.game_variables_prev = None
    self.pos = None
    self.pos_prev = None
    self.is_dead = None
    self.is_dead_prev = None
    self.step_this_episode = 0
    self.step_this_life = 0
    self._update_vars()

  # reset
  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)

    self.game_variables = None
    self.game_variables_prev = None
    self.pos = None
    self.pos_prev = None
    self.is_dead = None
    self.is_dead_prev = None
    self.step_this_episode = 0
    self.step_this_life = 0
    self._update_vars()

    return obs

  # step
  def reward(self, reward):
    self._update_vars()
    self._update_dead()
    self.step_this_episode += 1
    if not self.is_dead and self.is_dead_prev:
      self.step_this_life = 0
    else:
      self.step_this_life += 1

    r = 0. if self.is_recompute_reward else reward
    r += self._reward_living()
    # if self.is_dead: print('r = ', r)
    r += self._reward_dist()
    # if self.is_dead: print('r = ', r)
    r += self._reward_frag()
    # if self.is_dead: print('r = ', r)
    r += self._reward_health()
    # if self.is_dead: print('r = ', r)
    r += self._reward_ammo()
    # if self.is_dead: print('r = ', r)
    return r

  # helpers: updating
  def _update_vars(self):
    self.game_variables_prev = deepcopy(self.game_variables)
    if self.unwrapped._state is not None:  # ensure current frame is available
      game = self.unwrapped.game
      # common game variables
      self.game_variables = [
        game.get_game_variable(vd.GameVariable.FRAGCOUNT),
        game.get_game_variable(vd.GameVariable.HEALTH),
        game.get_game_variable(vd.GameVariable.SELECTED_WEAPON_AMMO),
      ]
      self.pos_prev = deepcopy(self.pos)
      self.pos = [
        game.get_game_variable(vd.GameVariable.POSITION_X),
        game.get_game_variable(vd.GameVariable.POSITION_Y),
      ]

  def _update_dead(self):
    self.is_dead_prev = deepcopy(self.is_dead)
    self.is_dead = self.unwrapped.game.is_player_dead()

  # helpers for reward
  def _reward_living(self):
    raise NotImplementedError

  def _reward_dist(self):
    raise NotImplementedError

  def _reward_frag(self):
    raise NotImplementedError

  def _reward_health(self):
    raise NotImplementedError

  def _reward_ammo(self):
    raise NotImplementedError


class RwdShapeWu2(RwdShapeWuBasic):
  """ Tweak of Wu Yuxin's reward shaping """

  # helpers for _reward
  def _reward_living(self):
    return -0.001

  def _reward_dist(self):
    ret = 0.
    if self.pos_prev is not None and self.step_this_life > self.LIFE_REAL_START:
      d = calc_dist(self.pos, self.pos_prev)
      # print(self.pos)
      # print(self.pos_prev)
      ret = 0.002 if d > self.dist_penalty_thres else 0.0
    return ret

  def _reward_frag(self):
    ret = 0.
    # unavailable for single player game; fine to keep it zero in this case
    if self.step_this_life > self.LIFE_REAL_START:
      r = (self.game_variables[self.IND_GAMEVAR_FRAG] -
           self.game_variables_prev[self.IND_GAMEVAR_FRAG])
      ret = float(r)
    return ret

  def _reward_health(self):
    ret = 0.
    if self.step_this_life > self.LIFE_REAL_START:
      r = (self.game_variables[self.IND_GAMEVAR_HEALTH] -
          self.game_variables_prev[self.IND_GAMEVAR_HEALTH])
      if r != 0:
        ret = 0.5 if r > 0 else -0.1
    return ret

  def _reward_ammo(self):
    ret = 0.
    if self.step_this_life > self.LIFE_REAL_START:
      r = (self.game_variables[self.IND_GAMEVAR_AMMO] -
           self.game_variables_prev[self.IND_GAMEVAR_AMMO])
      if r != 0:
        ret = 0.5 if r > 0 else -0.1
    return ret

class RwdShapeWu3(RwdShapeWuBasic):
  """ Tweak of Wu Yuxin's reward shaping """

  def __init__(self, env, 
          live,
          dist_inc, dist_dec,
          health_inc, health_dec,
          ammo_inc, ammo_dec):
    super(RwdShapeWu3, self).__init__(env)

    self._live = live
    self._dist_inc = dist_inc
    self._dist_dec = dist_dec
    self._health_inc = health_inc
    self._health_dec = health_dec
    self._ammo_inc = ammo_inc
    self._ammo_dec = ammo_dec
    
  # helpers for _reward
  def _reward_living(self):
    return self._live

  def _reward_dist(self):
    ret = 0.
    if self.pos_prev is not None and self.step_this_life > self.LIFE_REAL_START:
      d = calc_dist(self.pos, self.pos_prev)
      ret = self._dist_inc * d if d > self.dist_penalty_thres else self._dist_dec
    return ret

  def _reward_frag(self):
    ret = 0.
    # unavailable for single player game; fine to keep it zero in this case
    if self.step_this_life > self.LIFE_REAL_START:
      r = (self.game_variables[self.IND_GAMEVAR_FRAG] -
           self.game_variables_prev[self.IND_GAMEVAR_FRAG])
      ret = float(r)
    return ret

  def _reward_health(self):
    ret = 0.
    if self.step_this_life > self.LIFE_REAL_START:
      r = (self.game_variables[self.IND_GAMEVAR_HEALTH] -
          self.game_variables_prev[self.IND_GAMEVAR_HEALTH])
      if r != 0:
        ret = self._health_inc * r if r > 0 else self._health_dec * r
    return ret

  def _reward_ammo(self):
    ret = 0.
    if self.step_this_life > self.LIFE_REAL_START:
      r = (self.game_variables[self.IND_GAMEVAR_AMMO] -
           self.game_variables_prev[self.IND_GAMEVAR_AMMO])
      if r != 0:
        ret = self._ammo_inc * r if r > 0 else self._ammo_dec * r
    return ret
