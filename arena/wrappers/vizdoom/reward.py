from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from math import sqrt, cos, sin, pi

import gym
from gym.spaces import Box, Discrete
import vizdoom as vd
import numpy as np


class FragReward(gym.RewardWrapper):
  """ Frag as reward """
  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)
    self._prev_frag = 0
    self._cur_frag = 0

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self._prev_frag = self.env.unwrapped.game.get_game_variable(
      vd.GameVariable.FRAGCOUNT)
    self._cur_frag = self._prev_frag
    return obs

  def reward(self, r):
    self._cur_frag = self.env.unwrapped.game.get_game_variable(
      vd.GameVariable.FRAGCOUNT
    )
    rr = self._cur_frag - self._prev_frag
    self._prev_frag = self._cur_frag
    return float(rr)


class TrackerObjectDistAngleExampleReward(gym.RewardWrapper):
  """ Tracker-Object Distance-Angle Reward. Wrap over vec env.
  Presume GameVariables[0:3] are (x, y, angle), which can be done by config
  file, see:
  https://github.com/mwydmuch/ViZDoom/blob/6fe0d2470872adbfa5d18c53c7704e6ff103cacc/scenarios/health_gathering.cfg#L34
  https://github.com/mwydmuch/ViZDoom/blob/6fe0d2470872adbfa5d18c53c7704e6ff103cacc/examples/python/shaping.py#L82
  or by game.add_available_variables(...). See:
  https://github.com/mwydmuch/ViZDoom/blob/master/doc/Types.md#gamevariable
  https://github.com/mwydmuch/ViZDoom/blob/6fe0d2470872adbfa5d18c53c7704e6ff103cacc/examples/python/labels.py#L51
  For object tracking task. An example. """
  def __init__(self, venv):
    gym.RewardWrapper.__init__(self, venv)
    self._cur_pos_xya = None
    self._num_players = 2

  def reset(self, **kwargs):
    observations = self.env.reset(**kwargs)
    self._checkup()
    self._grab()
    return observations

  def step(self, actions):
    observations, _, dones, infos = self.env.step(actions)
    self._grab()
    rs = self.rewards_position_angle()
    return observations, rs, dones, infos

  def rewards_position_angle(self):
    def my_dist(xya_one, xya_two):
      dx, dy, da = [item1 - item2 for item1, item2 in zip(xya_one, xya_two)]
      return sqrt(dx*dx + dy*dy + da*da)

    if self._cur_pos_xya is None or len(self._cur_pos_xya) != self._num_players:
      return [0.0, 0.0]

    xya_tracker = self._cur_pos_xya[0]
    xya_object = self._cur_pos_xya[1]
    print('xya_tracker = ', xya_tracker)
    print('xya_object = ', xya_object)

    r_tracker = 1 / (my_dist(xya_tracker, xya_object) + 0.1)
    r_object = -r_tracker

    #return [1.0]
    return [r_tracker, r_object]

  def _checkup(self):
    assert(len(self.env.unwrapped.envs) == self._num_players)
    for e in self.env.unwrapped.envs:
      assert(len(e.unwrapped._state.game_variables) >= 3)  # at least x, y, a

  def _grab(self):
    self._cur_pos_xya = []
    for e in self.env.unwrapped.envs:
      this_state = e.unwrapped._state
      if this_state is not None:
        self._cur_pos_xya.append(this_state.game_variables)


class TrackerObjectDistAngleReward(gym.RewardWrapper):
  """ Tracker-Object Distance-Angle Reward. Wrap over vec env.

  Fangwei's settings. """
  def __init__(self, venv, max_steps):
    gym.RewardWrapper.__init__(self, venv)
    self._cur_pos_xya = None
    self._num_players = 2
    self.max_steps = max_steps

  def reset(self, **kwargs):
    observations = self.env.reset(**kwargs)
    self._checkup()
    self._grab()
    self.count_done = 0
    self.count_step = 0
    return observations

  def step(self, actions):
    self.count_step += 1

    observations, _, dones, infos = self.env.step(actions)
    self._grab()

    if all(dones):  # episode end
      return observations, [0.0, 0.0], dones, infos

    # print('tracker xya = ', self._cur_pos_xya[0])
    # print('object xya = ', self._cur_pos_xya[1])

    rs, outrange = self.rewards_position_angle()
    if outrange:
      self.count_done += 1
    else:
      self.count_done=0
    if self.count_done > 20 or self.count_step > self.max_steps:
      dones = [True, True]
    else:
      dones = [False, False]
    return observations, rs, dones, infos

  def world_to_local(self, xya_one, xya_two):
      # vizdoom fixed point angle to radius
      x0, y0, a0 = xya_one
      xt, yt, at = xya_two
      theta = a0 / 180.0 * pi
      # orientation to rotation
      theta -= pi/2
      # common origin of world and local coordinate system
      dx, dy = xt - x0, yt - y0
      # coordinate rotation
      x_ = dx * cos(theta) + dy * sin(theta)
      y_ = -dx * sin(theta) + dy * cos(theta)
      a_ = a0 - at
      return x_, y_, a_

  def get_reward(self, dx, dy, exp_dis=128):
    dist = sqrt(dx * dx + dy * dy)
    theta = abs(np.arctan2(dx, dy)/(pi/4))
    e_dis_relative = abs((dist - exp_dis)/exp_dis)
    reward_tracker = 1.0 - min(e_dis_relative, 1.0) - min(theta, 1.0)
    reward_tracker = max(reward_tracker, -1)
    e_theta = abs(theta - 1.0)
    e_dis = abs(e_dis_relative - 1.0)
    reward_object = 1.0 - (min(e_dis, 1.0) + min(e_theta, 1.0))
    outrange = False
    if e_dis_relative > 2 or theta > 1:
      outrange = True

    return reward_tracker, reward_object, outrange

  def rewards_position_angle(self):
    if self._cur_pos_xya is None:
      print ('None players!')
      return [0.0, 0.0], True
    if len(self._cur_pos_xya) == 0:
      print('0 players!')
      return [0.0, 0.0], True

    xya_object = self._cur_pos_xya[1]
    xya_tracker = self._cur_pos_xya[0]
    xx_, yy_, aa_ = self.world_to_local(xya_tracker, xya_object)
    r_tracker, r_object, outrange = self.get_reward(xx_, yy_, exp_dis=128)
    r_object = -r_tracker
    return [r_tracker, r_object], outrange

  def _checkup(self):
    assert(len(self.env.unwrapped.envs) == self._num_players)
    for e in self.env.unwrapped.envs:
      assert(len(e.unwrapped._state.game_variables) >= 3)  # at least x, y, a

  def _grab(self):
    self._cur_pos_xya = []
    for e in self.env.unwrapped.envs:
      this_state = e.unwrapped._state
      if this_state is not None:
        self._cur_pos_xya.append(this_state.game_variables)
