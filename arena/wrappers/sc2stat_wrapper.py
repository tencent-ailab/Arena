import logging

from gym import Wrapper
from pysc2.lib.typeenums import ABILITY_ID


class StatAllAction(Wrapper):
  """Statistics for all actions counting."""
  def __init__(self, env):
    super(StatAllAction, self).__init__(env)
    self._ab_dict = dict([(ab.value, 0) for ab in ABILITY_ID])

  def _reset_stat(self):
    self._ab_dict = dict([(ab.value, 0) for ab in ABILITY_ID])

  def _action_stat(self, actions):
    for action in actions[0]:
      if action.action_raw.unit_command.ability_id in self._ab_dict:
        self._ab_dict[action.action_raw.unit_command.ability_id] += 1

  def step(self, actions):
    self._action_stat(actions)
    obs, reward, done, info = self.env.step(actions)
    if done:
      for k, v in self._ab_dict.items():
        if v > 0:
          info[ABILITY_ID(k).name] = v
      self._reset_stat()
    return obs, reward, done, info


class StatZStatFn(Wrapper):
  """Statistics for ZStat Filename"""
  def step(self, actions):
    obs, reward, done, info = self.env.step(actions)
    if done:
      if not hasattr(self, 'inters'):
        logging.warning("Cannot find the field 'inters' for this env {}".format(str(self)))
        return obs, reward, done, info
      for ind, interf in enumerate(self.inters):
        key = 'agt{}zstat'.format(ind)
        root_interf = interf.unwrapped()
        if not hasattr(root_interf, 'cur_zstat_fn'):
          logging.warning("Cannot find the field 'cur_zstat_fn' for the root interface {}".format(root_interf))
          return obs, reward, done, info
        info[key] = root_interf.cur_zstat_fn
    return obs, reward, done, info