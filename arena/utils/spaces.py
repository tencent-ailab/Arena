import gym
from collections import Iterable


class NoneSpace(gym.Space):
    def __init__(self):
        super(NoneSpace, self).__init__(None, None)


class SC2RawObsSpace(NoneSpace):
    def __init__(self):
        super(SC2RawObsSpace, self).__init__()
        from pysc2.env.environment import TimeStep
        self.obs_class = TimeStep

    def sample(self):
        return self.obs_class([None] * len(self.obs_class._fields))

    def contains(self, x):
        return isinstance(x, self.obs_class)


class SC2RawActSpace(NoneSpace):
    def __init__(self):
        super(SC2RawActSpace, self).__init__()
        from s2clientprotocol.sc2api_pb2 import Action
        self.act_class = Action

    def sample(self):
        return []

    def contains(self, x):
        if isinstance(x, Iterable):
            return all([isinstance(a, self.act_class) for a in x])
        else:
            return isinstance(x, self.act_class)

