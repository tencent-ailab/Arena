""" Gym env wrappers """
from gym import spaces
import numpy as np

from arena.interfaces.interface import Interface


class NoopMaskInt(Interface):
    """ have to be wrapped after FullActInt or NoopActIntV2 which contains self.noop_cnt """
    def __init__(self, inter, max_noop_dim=10):
        super(self.__class__, self).__init__(inter)
        self.ability_dim = self.action_space.spaces[0].n
        self.noop_dim = max_noop_dim
        self.pre_obs_space = inter.observation_space

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def observation_space(self):
        obs_spec = spaces.Tuple(self.pre_obs_space.spaces + [
          spaces.Box(low=0, high=1, shape=(self.ability_dim,), dtype=np.bool),
          spaces.Box(low=0, high=1, shape=(self.noop_dim,), dtype=np.bool)])
        return obs_spec

    def obs_trans(self, raw_obs):
        if self.inter and hasattr(self.inter, 'noop_cnt'):
            obs = self.inter.obs_trans(raw_obs)
            obs += self._make_noop_mask(self.inter.noop_cnt)
            return obs
        else:
            raise BaseException('NoopInt has to be used together with '
                                'FullActInt or NoopActIntV2')

    def _make_noop_mask(self, noop_cnt):
        if noop_cnt > 0:
            ability_mask = np.zeros(shape=(self.ability_dim,), dtype=np.bool)
            ability_mask[0] = 1
            noop_mask = np.zeros(shape=(self.noop_dim,), dtype=np.bool)
            noop_mask[noop_cnt-1] = 1
        else:
            ability_mask = np.ones(shape=(self.ability_dim,), dtype=np.bool)
            noop_mask = np.ones(shape=(self.noop_dim,), dtype=np.bool)
        return ability_mask, noop_mask