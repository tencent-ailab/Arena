"""Vizdoom observation interfaces"""
from copy import deepcopy

import cv2
import numpy as np
import vizdoom as vd
from gym.spaces import Box, Tuple

from arena.interfaces.interface import Interface
from arena.utils.vizdoom.Rect import Rect
from arena.utils.vizdoom.player import PlayerConfig


class FrameVarObsInt(Interface):
    """Wu Yuxin's Observation as Screen + GameVariables.

    Expose observations as a list [screen, game_var], where
    screen.shape = (height, width, channel)
    and
    game_var.shape = (2,)
    which includes (health, ammo) normalized to [0.0, 1.0]
    """
    def __init__(self, inter, env, height=84, width=84):
        super(__class__, self).__init__(inter)
        self.env = env

        # export observation space
        channel = self.env.observation_space.shape[0]
        self.screen_sp = self.env.observation_space
        self.screen_sp.shape = (height, width, channel)

        self.game_var_sp = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self._height, self._width = height, width
        self._dft_gamevar = np.zeros(shape=self.game_var_sp.shape,
                                     dtype=self.game_var_sp.dtype)
        self._gamevar = np.array(deepcopy(self._dft_gamevar))

    @property
    def observation_space(self):
        return Tuple([Box(low=0.0, high=1.0, shape=self.screen_sp.shape, dtype=np.float32),
                      self.game_var_sp])

    @property
    def action_space(self):
        return self.env.action_space

    def obs_trans(self, frame):
        # Permute and resize
        frame = np.transpose(frame, axes=(1, 2, 0))
        frame = cv2.resize(frame, (self._height, self._width),
                           interpolation=cv2.INTER_AREA)
        # normalized frame
        frame = (np.array(frame) / 255)
        # normalized game vars
        self._grab_gamevar()
        return np.array([frame, self._gamevar])

    def reset(self, obs, **kwargs):
        super(FrameVarObsInt, self).reset(obs, **kwargs)
        # self.wrapper = FrameVarObsFunc(obs, self.env, self.use_attr)

    def _grab_gamevar(self):
        if self.env.unwrapped._state is not None:
            game = self.env.unwrapped.game
            self._gamevar[0] = game.get_game_variable(vd.GameVariable.HEALTH) / 100.0
            self._gamevar[1] = game.get_game_variable(vd.GameVariable.AMMO2) / 15.0

    def _get_available_game_variables_dim(player_cfg):
        g = vd.DoomGame()
        g = player_setup(g, player_cfg)
        return len(g.get_available_game_variables())


class ReshapedFrameObsInt(Interface):
    """Reshaped Frame as Observation.

    TODO(pengsun): more descriptions."""
    def __init__(self, inter, env, height=168, width=168):
        super(__class__, self).__init__(inter)
        self.env = env
        # export observation space
        channel = self.env.observation_space.shape[0]*2
        center_patch = 0.3
        frac = center_patch / 2
        W, H = 800, 450
        self._center_rect = Rect(*map(int,
                                      [W / 2 - W * frac, H / 2 - H * frac, W * frac * 2, H * frac * 2]))
        self.screen_sp = self.env.observation_space
        self.screen_sp.shape = (height, width, channel)
        self._height, self._width = height, width

    @property
    def observation_space(self):
        return Box(low=0.0, high=1.0, shape=self.screen_sp.shape, dtype=np.float32)

    @property
    def action_space(self):
        return self.env.action_space

    def obs_trans(self, frame):
        if frame.shape != (168, 168, 6):
            # Permute and resize
            frame = np.transpose(frame, axes=(1, 2, 0))
            ## normalized frame
            frame = (np.array(frame) / 255.0)
            center_patch = self._center_rect.roi(frame)
            frame = cv2.resize(frame, (self._height, self._width),
                               interpolation=cv2.INTER_AREA)
            center_patch = cv2.resize(center_patch, (self._height, self._width),
                                      interpolation=cv2.INTER_AREA)
            frame = np.concatenate((frame, center_patch), axis=2)
            ## normalized frame
            # frame = (np.array(frame)/255)
        return frame

    def reset(self, obs, **kwargs):
        super(ReshapedFrameObsInt, self).reset(obs, **kwargs)
