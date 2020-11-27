"""This file contains the observation interfaces for pommerman."""

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
    """This is the observation transformation function used in BoardMapObs,
      it extracted image feature of obs

    :param obs: origin obs
    :param items: items as channels
    :param override: override the previous obs or append
    :param space_old: observation space of previous iterface
    """

    def __init__(self, obs, items, use_attr, override, space_old):
        self.override = override
        self.use_attr = use_attr
        self.items = list(items)
        self.shape = list(obs['board'].shape) + [len(self.items) + 5 + self.use_attr * 4]
        observation_space = spaces.Box(0.0, float('inf'), self.shape, dtype=np.float32)
        if self.override or isinstance(space_old, NoneSpace):
            self.observation_space = spaces.Tuple((observation_space,))
        else:
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (observation_space,))

    def observation_transform(self, obs_pre, obs):
        board = obs['board']
        new_obs = [board == i for i in self.items]
        def expand_bomb_blast(board):
            new_board = np.zeros(board.shape)
            for i in range(board.shape[0]):
                for j in range(board.shape[1]):
                    if board[i][j] == 0:
                        continue
                    s = int(board[i][j] - 1)
                    new_board[range(max(0, i-s), min(11, i+s+1)), j] = 1
                    new_board[i, range(max(0, j-s), min(11, j+s+1))] = 1
            return new_board
        new_obs.append(expand_bomb_blast(obs['bomb_blast_strength']))
        def expand_bomb_life(obs):
            new_board = np.array(obs['bomb_life'])
            max_x, max_y = new_board.shape
            bomb_pos = np.nonzero(new_board)
            bomb_life = new_board[bomb_pos]
            for i in np.argsort(-bomb_life):
                x, y = bomb_pos[0][i], bomb_pos[1][i]
                s = int(obs['bomb_blast_strength'][x, y] - 1)
                normed_life = (10 - bomb_life[i]) / 10.0
                new_board[range(max(0, x-s), min(max_x, x+s+1)), y] = normed_life
                new_board[x, range(max(0, y-s), min(max_y, y+s+1))] = normed_life
            return new_board
        new_obs.append(expand_bomb_life(obs))

        # add my position, one-hot image
        position = np.zeros(board.shape)
        position[obs['position']] = 1
        new_obs.append(position)

        # add teammate
        if obs['teammate'] is not None:
            new_obs.append(board == obs['teammate'].value)
        else:
            new_obs.append(np.zeros(board.shape))

        # add enemies
        enemies = [board == e.value for e in obs['enemies']]
        new_obs.append(np.any(enemies, axis=0))

        if self.use_attr:
            new_obs.append(np.full(board.shape, obs['ammo']/3.0))
            new_obs.append(np.full(board.shape, obs['blast_strength'] / 5.0))
            new_obs.append(np.full(board.shape, obs['can_kick']))
            new_obs.append(np.full(board.shape, board[obs['position']] in obs['alive']))

        new_obs = np.stack(new_obs, axis=2)
        return (new_obs,) if self.override else obs_pre + (new_obs,)


class BoardMapObs(AppendObsInt):
    """Observation interface to append a image-like feature.
      It extracted 0-1 maps of items in self.items.
      Also extend bomb_blast_strength and bomb_life
      from the position of bomb to explosion range. """

    def __init__(self, inter, use_attr=True, override=True):
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

    def build_wrapper(self):
        teammate_id = self.unwrapped()._obs['teammate'].value
        self_id = (teammate_id - 8) % 4 + 10
        enemy_ids = [e.value for e in self.unwrapped()._obs['enemies']]
        items = self.items + (self_id, teammate_id) + tuple(enemy_ids)
        self.wrapper = BoardMapObsFunc(self.unwrapped()._obs, items,
                                       self.use_attr, self.override,
                                       space_old=self.inter.observation_space)


class AttrObsInt(AppendObsInt):
    """Observation interface to append a vector feature of bomber's attributes.
       Including is_alive, pos (x, y),  blast_strength, can_kick, ammo"""

    class AttrFunc(object):
        def __init__(self, obs, override, space_old):
            self._attr_dim = 6
            self._board_shape = obs['board'].shape
            self.id = (obs['teammate'].value - 8) % 4 + 10
            obs_space = spaces.Box(low=-1, high=2, dtype=np.float32,
                                   shape=(self._attr_dim,))
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((obs_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (obs_space,))

        def observation_transform(self, obs_pre, obs):
            units_vec = []
            pos = obs['position']
            alive = self.id in obs['alive']
            units_vec.append(alive)
            units_vec.append(pos[0] / (self._board_shape[0] - 1) - 0.5)
            units_vec.append(pos[1] / (self._board_shape[1] - 1) - 0.5)
            if alive:
                units_vec.extend([obs['blast_strength'] / 5.0,
                                  obs['can_kick'],
                                  obs['ammo'] / 3.0])
            else:
                units_vec.extend([0., 0., 0.])
            observation = [np.array(units_vec, dtype=np.float32)]
            return (observation,) if self.override else obs_pre + (observation,)

    def build_wrapper(self):
        self.wrapper = self.AttrFunc(self.unwrapped()._obs, override=self.override,
                                     space_old=self.inter.observation_space)


class PosObsInt(AppendObsInt):
    class PosFunc(object):
        def __init__(self, obs, override, space_old):
            self._attr_dim = 2
            obs_space = spaces.Box(low=-1, high=2, dtype=np.int32,
                                   shape=(self._attr_dim,))
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((obs_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (obs_space,))

        def observation_transform(self, obs_pre, obs):
            observation = np.array(obs['position'], dtype=np.int32)
            return (observation,) if self.override else obs_pre + (observation,)

    def reset(self, obs, **kwargs):
        super(PosObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.PosFunc(self.unwrapped()._obs, override=self.override,
                                    space_old=self.inter.observation_space)


class ActMaskObsInt(AppendObsInt):
    """Observation interface to append the availability of each action."""

    class ActMaskFunc(object):
        def __init__(self, obs, space_old):
            self.id = (obs['teammate'].value - 8) % 4 + 10
            self.shape = obs['board'].shape
            obs_space = spaces.Box(low=-1, high=2, dtype=np.bool, shape=(6,))
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (obs_space,))
            self.pathing_items = [Item.Passage.value, Item.ExtraBomb.value,
                                  Item.IncrRange.value, Item.Kick.value]

        def in_board(self, pos):
            return (0 <= pos[0] < self.shape[0]) and (0 <= pos[1] < self.shape[1])

        def observation_transform(self, obs_pre, obs):
            act_mask = np.zeros((6,))
            pos = obs['position']
            act_mask[0] = 1
            if self.id in obs['alive']:
                for j, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    new_pos = (pos[0] + dx, pos[1] + dy)
                    if self.in_board(new_pos):
                        if obs['board'][new_pos] in self.pathing_items:
                            act_mask[j + 1] = 1
                        elif obs['can_kick'] and obs['board'][new_pos] == Item.Bomb.value:
                            further_pos = (pos[0] + 2 * dx, pos[1] + 2 * dy)
                            if (self.in_board(further_pos) and
                                obs['board'][further_pos] in self.pathing_items):
                                act_mask[j + 1] = 1
            act_mask[-1] = obs['ammo'] > 0 and obs['bomb_blast_strength'][pos] == 0
            return obs_pre + (act_mask.reshape([-1]),)

    def build_wrapper(self):
        self.wrapper = self.ActMaskFunc(self.unwrapped()._obs,
                                        space_old=self.inter.observation_space)


class RotateInt(Interface):
    '''Observation interface to rotate the board (and action) according to agent_id.
       After rotation, the agent will be born at the left-top corner.
       This interface is ported from 2018 NeurIPS NavocadoAgent.
    '''

    def __init__(self, inter):
        super(RotateInt, self).__init__(inter)
        self.agent_id = None
        self.board_shape = (11, 11)
        self._action_map = {
            10: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
            11: {0: 0, 1: 3, 2: 4, 3: 2, 4: 1, 5: 5},
            12: {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 5},
            13: {0: 0, 1: 4, 2: 3, 3: 1, 4: 2, 5: 5},
        }

    def reset(self, obs, **kwargs):
        super(RotateInt, self).reset(obs)
        self.board_shape = obs['board'].shape
        self.agent_id = obs['board'][tuple(obs['position'])]
        self.action_map = self._action_map[self.agent_id]
        return self._obs_trans(obs)

    def _obs_trans(self, obs):
        #print(obs['board'])
        norm_obs = {}
        bias = self.agent_id - 10
        mat_props = ['board', 'bomb_blast_strength', 'bomb_life']
        arr_props = ['alive', 'teammate', 'enemies']
        unchange_props = ['blast_strength', 'can_kick', 'ammo',
                          'game_type', 'game_env', 'step_count']
        for m in unchange_props:
            norm_obs[m] = obs[m]
        for m in mat_props:
            if m == 'board':
                norm_obs[m] = self._rotate(obs[m], bias, True)
            else:
                norm_obs[m] = self._rotate(obs[m], bias, False)
        for m in arr_props:
            norm_obs[m] = self._shift(obs[m], bias)
        norm_obs['position'] = tuple(self._rotate_pos(obs['position'], bias))
        self.unwrapped()._obs = norm_obs
        #print(norm_obs['board'])
        return norm_obs

    def _act_trans(self, act):
        return self.action_map[act]

    def _rotate(self, mat, bias, change_id=False):
        if bias == 0:
            rot_mat = np.copy(mat)
            return rot_mat
        rot_mat = np.copy(np.rot90(mat, k=4 - bias))
        if change_id:
            agent_list = np.argwhere(rot_mat > 9)
            for agent in agent_list:
                agent = tuple(agent)
                rot_mat[agent] = (rot_mat[agent] - bias - 10) % 4 + 10
        return rot_mat

    def _shift(self, arr, bias):
        if bias == 0:
            new_arr = copy.copy(arr)
            return new_arr
        if type(arr) == list:
            new_arr = []
            for a in arr:
                if type(a) == int:
                    if a == 9:
                        new_arr.append(a)
                    else:
                        new_arr.append((a - bias - 10) % 4 + 10)
                else:
                    if a.value == 9:
                        new_arr.append(a)
                    else:
                        new_arr.append(Item((a.value - bias - 10) % 4 + 10))
            try:
                new_arr = sorted(new_arr)
            except:
                new_arr = sorted(new_arr, key=lambda x: x.value if x.value != 9 else 20)
        elif type(arr) == int:
            if arr == 9:
                new_arr = arr
            else:
                new_arr = (arr - bias - 10) % 4 + 10
        else:
            if arr.value == 9:
                new_arr = arr
            else:
                new_arr = Item((arr.value - bias - 10) % 4 + 10)
        return new_arr

    def _rotate_pos(self, position, bias):
        if bias == 0:
            return position
        elif bias == 1:
            return (position[1],
                    self.board_shape[0] - 1 - position[0])
        elif bias == 2:
            return (self.board_shape[0] - 1 - position[0],
                    self.board_shape[1] - 1 - position[1])
        elif bias == 3:
            return (self.board_shape[1] - 1 - position[1],
                    position[0])
