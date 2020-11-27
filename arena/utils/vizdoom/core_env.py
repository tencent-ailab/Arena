from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Box
from gym.spaces import Tuple
import vizdoom as vd

from arena.utils.vizdoom.run_parallel import RunParallel
from arena.utils.vizdoom.player import player_setup, player_host_setup, player_join_setup


def _get_screen_shape(vd_resolution, vd_format):
  tmp = {
    (vd.ScreenFormat.CBCGCR, vd.ScreenResolution.RES_160X120): (3, 120, 160),
    (vd.ScreenFormat.CBCGCR, vd.ScreenResolution.RES_640X360): (3, 640, 360),
    (vd.ScreenFormat.CBCGCR, vd.ScreenResolution.RES_800X450): (3, 800, 450),
    (vd.ScreenFormat.CBCGCR, vd.ScreenResolution.RES_800X600): (3, 800, 600),
  }
  return tmp[(vd_format, vd_resolution)]


def _get_action_dim(player_cfg):
  g = vd.DoomGame()
  g = player_setup(g, player_cfg)
  return len(g.get_available_buttons())


def _get_available_game_variables_dim(player_cfg):
    g = vd.DoomGame()
    g = player_setup(g, player_cfg)
    return len(g.get_available_game_variables())


class PlayerEnv(gym.Env):
  # TODO(pengsun): delete/trim this class
  def __init__(self, cfg):
    self.cfg = cfg
    self.game = None

    # export observation space & action space
    self.observation_space = gym.spaces.Box(
      low=0, high=255, dtype=np.uint8,
      shape=_get_screen_shape(cfg.screen_resolution, cfg.screen_format)
    )
    self.action_space = gym.spaces.Box(low=0, high=1, dtype=np.float32,
                                       shape=(_get_action_dim(cfg),))
    # export predefined actions
    # self.action_noop = np.zeros(shape=self.action_space.shape)
    # self.action_noop[0] = 1
    self.action_noop = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    self.action_use = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    self.action_fire = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    self._state = None
    self._obs = None
    self._rwd = None
    self._done = None
    self._act = None

  def reset(self):
    if not self.cfg.is_multiplayer_game:
      if self.game is None:
        self._init_game()
      self.game.new_episode()
    else:
      self._init_game()
      if self.cfg.num_bots > 0:
        self._add_bot()

    self._state, self._obs, self._done = self._grab()
    return self._obs

  def step(self, action):
    self._rwd = self.game.make_action(action, self.cfg.repeat_frame)
    self._state, self._obs, self._done = self._grab()
    return self._obs, self._rwd, self._done, {}

  def close(self):
    if self.game:
      self.game.close()

  def render(self, *args):
    return self._obs

  def _init_game(self):
    self.close()

    game = vd.DoomGame()
    game = player_setup(game, self.cfg)

    if self.cfg.is_multiplayer_game:
      if self.cfg.host_cfg is not None:
        game = player_host_setup(game, self.cfg.host_cfg)
      elif self.cfg.join_cfg is not None:
        game = player_join_setup(game, self.cfg.join_cfg)
      else:
        raise ValueError('neither host nor join, error!')


    game.init()
    self.game = game

  def _grab(self):
    state = self.game.get_state()
    done = self.game.is_episode_finished()
    if done:
      obs = np.ndarray(shape=self.observation_space.shape,
                       dtype=self.observation_space.dtype)
    else:
      obs = state.screen_buffer
    return state, obs, done

  def _add_bot(self):
    self.game.send_game_command("removebots")
    for i in range(self.cfg.num_bots):
      self.game.send_game_command("addbot")


class VecEnv(gym.Env):
  # TODO(pengsun): delete/trim this class
  def __init__(self, envs):
    # export observation space & action space
    self.observation_space = [e.observation_space for e in envs]
    self.action_space = [e.action_space for e in envs]

    self._envs = envs
    self._par = RunParallel()

  def reset(self):
    observations = self._par.run((e.reset) for e in self._envs)
    return observations

  def step(self, actions):
    ret = self._par.run((e.step, act)
                        for e, act in zip(self._envs, actions))
    observations, rewards, dones, infos = [item for item in zip(*ret)]
    return observations, rewards, dones, infos

  def close(self):
    self._par.run((e.close) for e in self._envs)

  def render(self, *args):
    obs = self._par.run((e.render) for e in self._envs)
    return obs

  @property
  def envs(self):
    return self._envs
