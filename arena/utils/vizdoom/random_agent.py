from random import choice


class RandomAgent(object):
  def __init__(self, allowed_action):
    self._allowed_action = allowed_action

  def step(self, obs):
    return choice(self._allowed_action)
