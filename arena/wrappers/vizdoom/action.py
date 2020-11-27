from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import gym
from gym.spaces import Box, Discrete


class Discrete7Action(gym.ActionWrapper):
  """ Discrete 7 Actions """
  def __init__(self, env):
    gym.ActionWrapper.__init__(self, env)

    self.action_space = Discrete(n=7)

    self._allowed_action = [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]

  def action(self, a):
    return self._allowed_action[a]

  def reverse_action(self, action):
    raise NotImplementedError


class Discrete3MoveAction(gym.ActionWrapper):
  """ Discrete 3 Actions, just for moving """
  def __init__(self, env):
    gym.ActionWrapper.__init__(self, env)

    self.action_space = Discrete(n=3)

    self._allowed_action = [
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]

  def action(self, a):
    return self._allowed_action[a]

  def reverse_action(self, action):
    raise NotImplementedError


class Discrete6MoveAction(gym.ActionWrapper):
  """ Discrete 6 Actions, just for moving """
  def __init__(self, env):
    gym.ActionWrapper.__init__(self, env)

    self.action_space = Discrete(n=6)

    self._allowed_action = [
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # TURN_LEFT
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # TURN_RIGHT
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # MOVE_RIGHT
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # MOVE_LEFT
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # MOVE_FORWARD
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # MOVE_BACKWARD
    ]

  def action(self, a):
    return self._allowed_action[a]

  def reverse_action(self, action):
    raise NotImplementedError

class Discrete5MoveAction(gym.ActionWrapper):
  """ Discrete 6 Actions, just for moving """
  def __init__(self, env):
    gym.ActionWrapper.__init__(self, env)

    self.action_space = Discrete(n=5)

    self._allowed_action = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No opt
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # TURN_LEFT
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # TURN_RIGHT
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # MOVE_FORWARD
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # MOVE_BACKWARD
    ]

  def action(self, a):
    return self._allowed_action[a]

  def reverse_action(self, action):
    raise NotImplementedError

class Discrete7MoveAction(gym.ActionWrapper):
  """ Discrete 6 Actions, just for moving """
  def __init__(self, env):
    gym.ActionWrapper.__init__(self, env)

    self.action_space = Discrete(n=7)

    self._allowed_action = [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No opt
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # TURN_LEFT
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # TURN_RIGHT
      [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # TURN_LEFT + MOVE_FORWARD
      [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],  # TURN_RIGHT + MOVE_FORWARD
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # MOVE_FORWARD
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # MOVE_BACKWARD
    ]

  def action(self, a):
    return self._allowed_action[a]

  def reverse_action(self, action):
    raise NotImplementedError
