from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from arena.interfaces.interface import Interface
from arena.interfaces.common import AppendObsInt
from arena.utils.spaces import NoneSpace
from gym import spaces
from pommerman.constants import Item


class BoardMapObsFunc(object):
    def __init__(self, obs, items, override, use_attr, space_old):
        self.override = override
        self.use_attr = use_attr
        self.items = list(items)
        self.items.extend(obs['teammate'] + obs['enemies'])
        self.shape = list(obs['board'].shape) + [len(self.items) + 2 + 3 * use_attr]
        self.item_map = {v:k for k, v in enumerate(self.items)}
        observation_space = spaces.Box(0.0, float('inf'), self.shape, dtype=np.float32)
        if self.override or isinstance(space_old, NoneSpace):
            self.observation_space = spaces.Tuple((observation_space,))
        else:
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (observation_space,))

    def observation_transform(self, obs_pre, obs):
        new_obs = np.zeros(self.shape, dtype=np.float32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                k = obs['board'][i, j]
                if k in self.item_map:
                    new_obs[i, j, self.item_map[k]] = 1
        def expand_bomb_blast(board):
            new_board = np.zeros(board.shape)
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board[i][j] == 0:
                        continue
                    s = int(board[i][j] - 1)
                    for ii in range(max(0, i-s), min(11, i+s+1)):
                        new_board[ii, j] = 1
                    for jj in range(max(0, j-s), min(11, j+s+1)):
                        new_board[i, jj] = 1
            return new_board
        new_obs[:, :, -(2 + 3 * self.use_attr)] = expand_bomb_blast(obs['bomb_blast_strength'])
        def expand_bomb_life(obs):
            new_board = np.array(obs['bomb_life'])
            max_x, max_y = new_board.shape
            bomb_pos = np.nonzero(new_board)
            bomb_life = new_board[bomb_pos]
            for i in np.argsort(bomb_life):
                x, y = bomb_pos[0][i], bomb_pos[1][i]
                s = int(obs['bomb_blast_strength'][x, y] - 1)
                for ii in range(max(0, x-s), min(max_x, x+s+1)):
                    new_board[ii, y] = new_board[x, y]
                for jj in range(max(0, y-s), min(max_y, y+s+1)):
                    new_board[x, jj] = new_board[x, y]
            for i in range(max_x):
                for j in range(max_y):
                    if new_board[i, j] != 0:
                        new_board[i, j] = (10 - new_board[i, j]) / 10.0
            return new_board
        new_obs[:, :, -(1 + 3 * self.use_attr)] = expand_bomb_life(obs)

        if self.use_attr:
            for i, pos in enumerate(obs['position']):
                if obs['alive'][i]:
                    new_obs[pos[0], pos[1], -3] = obs['blast_strength'][i] / 5.0
                    new_obs[pos[0], pos[1], -2] = obs['can_kick'][i]
                    new_obs[pos[0], pos[1], -1] = obs['ammo'][i] / 3.0
        return (new_obs,) if self.override else obs_pre + (new_obs,)


class BoardMapObs(AppendObsInt):
    def __init__(self, inter, override=True, use_attr=True):
        super(BoardMapObs, self).__init__(inter, override)
        self.items = (Item.Rigid.value,
                      Item.Wood.value,
                      Item.Bomb.value,
                      Item.Flames.value,
                      Item.Fog.value,
                      Item.ExtraBomb.value,
                      Item.IncrRange.value,
                      Item.Kick.value)
        self.use_attr = use_attr

    def reset(self, obs, **kwargs):
        super(BoardMapObs, self).reset(obs, **kwargs)
        self.wrapper = BoardMapObsFunc(self.unwrapped()._obs, self.items,
                                       self.override, self.use_attr,
                                       space_old=self.inter.observation_space)


class CombineObsInt(Interface):
    def __init__(self, inter, remove_dead_view = True):
        super(CombineObsInt, self).__init__(inter)
        self.remove_dead_view = remove_dead_view

    def reset(self, obs, **kwargs):
        super(CombineObsInt, self).reset(obs, **kwargs)
        self.obs_trans(obs)

    def obs_trans(self, obs):
        state1, state2 = obs
        if self.remove_dead_view and state2['teammate'].value not in state2['alive']:
            state = copy.deepcopy(state2)
        elif self.remove_dead_view and  state1['teammate'].value not in state1['alive']:
            state = copy.deepcopy(state1)
        else:
            state = copy.deepcopy(state1)
            for i in range(state1['board'].shape[0]):
                for j in range(state1['board'].shape[1]):
                    if (state1['board'][i, j] == Item.Fog.value and
                            state2['board'][i, j] != Item.Fog.value):
                        state['board'][i, j] = state2['board'][i, j]
        state['position'] = (state1['position'], state2['position'])
        state['blast_strength'] = (state1['blast_strength'], state2['blast_strength'])
        state['can_kick'] = (state1['can_kick'], state2['can_kick'])
        state['ammo'] = (state1['ammo'], state2['ammo'])
        state['teammate'] = (state2['teammate'].value, state1['teammate'].value)
        state['enemies'] = (state['enemies'][0].value, state['enemies'][1].value)
        state['alive'] = [t in state['alive'] for t in state['teammate']] + \
                         [e in state['alive'] for e in state['enemies']]
        self.unwrapped()._obs = state
        return state


class AttrObsInt(AppendObsInt):
    class AttrFunc(object):
        def __init__(self, obs, override, space_old):
            self._attr_dim = 6
            self._board_shape = obs['board'].shape
            obs_space = spaces.Box(low=-1, high=2, dtype=np.float32,
                                   shape=(self._attr_dim * len(obs['position']),))
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((obs_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (obs_space,))

        def observation_transform(self, obs_pre, obs):
            units_vec = []
            for i, pos in enumerate(obs['position']):
                alive = obs['alive'][i]
                units_vec.append(alive)
                units_vec.append(pos[0] / (self._board_shape[0] - 1) - 0.5)
                units_vec.append(pos[1] / (self._board_shape[1] - 1) - 0.5)
                if alive:
                    units_vec.extend([obs['blast_strength'][i] / 5.0,
                                      obs['can_kick'][i],
                                      obs['ammo'][i] / 3.0])
                else:
                    units_vec.extend([0., 0., 0.])
            observation = np.array(units_vec, dtype=np.float32)
            return (observation,) if self.override else obs_pre + (observation,)

    def reset(self, obs, **kwargs):
        super(AttrObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.AttrFunc(self.unwrapped()._obs, override=self.override,
                                     space_old=self.inter.observation_space)


class PosObsInt(AppendObsInt):
    class PosFunc(object):
        def __init__(self, obs, override, space_old):
            self._attr_dim = 2 * len(obs['position'])
            obs_space = spaces.Box(low=-1, high=2, dtype=np.int32,
                                   shape=(self._attr_dim,))
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((obs_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (obs_space,))

        def observation_transform(self, obs_pre, obs):
            units_vec = []
            for i, pos in enumerate(obs['position']):
                units_vec.append(pos[0])
                units_vec.append(pos[1])
            observation = np.array(units_vec, dtype=np.int32)
            return (observation,) if self.override else obs_pre + (observation,)

    def reset(self, obs, **kwargs):
        super(PosObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.PosFunc(self.unwrapped()._obs, override=self.override,
                                    space_old=self.inter.observation_space)


class ActMaskObsInt(AppendObsInt):
    class ActMaskFunc(object):
        def __init__(self, obs, space_old):
            self.shape = obs['board'].shape
            obs_space = spaces.Box(low=-1, high=2, dtype=np.int32,
                                   shape=(6*2,))
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (obs_space,))
            self.pathing_items = [Item.Passage.value, Item.ExtraBomb.value,
                                  Item.IncrRange.value, Item.Kick.value]

        def in_board(self, pos):
            return (0 <= pos[0] < self.shape[0]) and (0 <= pos[1] < self.shape[1])

        def observation_transform(self, obs_pre, obs):
            act_mask = np.zeros((2, 6))
            for i, pos in enumerate(obs['position'][0:2]):
                act_mask[i, 0] = 1
                if not obs['alive'][i]:
                    continue
                for j, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if self.in_board(new_pos):
                        if obs['board'][new_pos] in self.pathing_items:
                            act_mask[i, j + 1] = 1
                        elif obs['can_kick'][i] and obs['board'][new_pos] == Item.Bomb.value:
                            further_pos = (pos[0] + 2 * dx, pos[1] + 2 * dy)
                            if (self.in_board(further_pos) and
                                obs['board'][further_pos] in self.pathing_items):
                                act_mask[i, j + 1] = 1
                act_mask[i, -1] = obs['ammo'][i] > 0 and obs['bomb_blast_strength'][pos] == 0
            return obs_pre + (act_mask.reshape([-1]),)

    def reset(self, obs, **kwargs):
        super(ActMaskObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.ActMaskFunc(self.unwrapped()._obs,
                                        space_old=self.inter.observation_space)