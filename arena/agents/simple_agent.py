"""
Test agent to control marines
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from arena.agents.base_agent import BaseAgent
import gym


class RandomAgent(BaseAgent):
    """Random action agent."""
    def __init__(self, action_space=None):
        super(RandomAgent, self).__init__()
        self.action_space = action_space

    def step(self, obs):
        super(RandomAgent, self).step(obs)
        if hasattr(self.action_space, 'sample'):
            return self.action_space.sample()
        else:
            return None

    def reset(self, timestep=None):
        super(RandomAgent, self).reset(timestep)
        assert isinstance(self.action_space, gym.Space)
