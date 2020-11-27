from gym import spaces
from arena.utils.spaces import NoneSpace


class RawInt(object):
    """ This interface is usually used at the env side """

    def __init__(self):
        self._observation_space = NoneSpace()
        self._action_space = NoneSpace()
        self.steps = None

    def setup(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    def reset(self, obs, **kwargs):
        self._obs = obs
        self.steps = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def obs_trans(self, obs):
        self._obs = obs
        return obs

    def act_trans(self, act):
        self._act = act
        self.steps += 1
        return act

    def __str__(self):
        return '<'+self.__class__.__name__+'>'

    def unwrapped(self):
        return self
