"""This file contains the observation interfaces for dm-control soccer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from arena.interfaces.interface import Interface
from arena.interfaces.common import AppendObsInt
from arena.utils.spaces import NoneSpace
from gym import spaces

class Dict2Vec(Interface):
    @property
    def observation_space(self):
        if isinstance(self.inter.observation_space, NoneSpace):
            return NoneSpace()
        assert isinstance(self.inter.observation_space, spaces.Dict)
        return self.convert_OrderedDict(self.inter.observation_space.spaces)

    def convert_OrderedDict(self, odict):
        # concatentation
        numdim = sum([np.int(np.prod(odict[key].shape)) for key in odict])
        return spaces.Box(-np.inf, np.inf, shape=(numdim,))

    def convert_observation(self, dict_obs):
        numdim = sum([np.int(np.prod(dict_obs[key].shape)) for key in dict_obs])
        space_obs = np.zeros((numdim,))
        i = 0
        for key in dict_obs:
            space_obs[i:i+np.prod(dict_obs[key].shape)] = dict_obs[key].ravel()
            i += np.prod(dict_obs[key].shape)
        return space_obs

    def obs_trans(self, obs):
        ret =  self.convert_observation(obs)
        return ret

class ConcatObsAct(Interface):
    @property
    def observation_space(self):
        if isinstance(self.inter.observation_space, NoneSpace):
            return NoneSpace()
        assert isinstance(self.inter.observation_space, spaces.Tuple)
        sps = self.inter.observation_space.spaces
        if any([isinstance(sp, NoneSpace) for sp in sps]):
            return NoneSpace()
        numdim = sum([ np.int(np.prod(sps[i].shape)) for i in range(len(sps))])
        return spaces.Box(-np.inf, np.inf, shape=(numdim,))

    @property
    def action_space(self):
        if isinstance(self.inter.action_space, NoneSpace):
            return NoneSpace()
        assert isinstance(self.inter.action_space, spaces.Tuple)
        sps = self.inter.action_space.spaces
        if any([isinstance(sp, NoneSpace) for sp in sps]):
            return NoneSpace()
        numdim = sum([ np.int(np.prod(sps[i].shape)) for i in range(len(sps))])
        return spaces.Box(-1., 1., shape=(numdim,))

    def _obs_trans(self, obs):
        return np.concatenate(obs)

    def obs_trans(self, obs):
        """ Observation Transformation. This is a recursive call. """
        obs = self.inter.obs_trans(obs)
        return self._obs_trans(obs)

    def _act_trans(self, acts):
        rets = []
        sps = self.inter.action_space.spaces
        i = 0
        for agsps in sps:
            size = agsps.shape[0]
            rets.append(acts[i:i+size])
            i += size
        return rets