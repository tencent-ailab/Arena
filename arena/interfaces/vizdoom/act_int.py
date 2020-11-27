"""Vizdoom action interfaces"""
from copy import deepcopy

from gym.spaces import Discrete as GymDiscrete
from vizdoom import Button

from arena.interfaces.interface import Interface


class Discrete6ActionInt(Interface):
  """Wu Yuxing's 6-action setting.

  The six actions are:
    move forward,
    fire,
    move left,
    move right,
    turn left,
    turn right,
  """
  def __init__(self, inter):
    super(Discrete6ActionInt, self).__init__(inter)

    self.allowed_buttons = [
      Button.ATTACK,
      Button.TURN_LEFT,
      Button.TURN_RIGHT,
      Button.MOVE_RIGHT,
      Button.MOVE_LEFT,
      Button.MOVE_FORWARD,
      Button.MOVE_BACKWARD,
      Button.SPEED,
      Button.TURN180,
      Button.TURN_LEFT_RIGHT_DELTA
    ]
    # NOTE: [7:] actions are not exposed
    self.allowed_actions = [
      [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 0 move fast forward
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 fire
      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 2 move left
      [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 3 move right
      [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 4 turn left
      [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 5 turn right
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 20],  # 6 turn left 40 degree and move forward
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 20],  # 7 turn right 40 degree and move forward
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8 move forward
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9 turn 180
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10 move left
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11 move right
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 turn left
      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13 turn right
    ]
    pass

  @property
  def action_space(self):
    return GymDiscrete(n=6)

  def act_trans(self, act):
    # act (as index) -> button vector
    act = int(act)
    return self.allowed_actions[act]
