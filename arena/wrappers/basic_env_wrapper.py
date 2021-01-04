from gym import ObservationWrapper
from gym import Wrapper, RewardWrapper
from gym import spaces
from itertools import chain
from arena.utils.spaces import NoneSpace
from random import random
import numpy as np


class RandPlayer(Wrapper):
  """ Wrapper for randomizing players order """

  def __init__(self, env):
    super(RandPlayer, self).__init__(env)
    assert len(self.env.action_space.spaces) == 2
    assert self.env.action_space.spaces[0] == self.env.action_space.spaces[1]
    assert self.env.observation_space.spaces[0] == self.env.observation_space.spaces[1]
    self.change_player = random() < 0.5

  def reset(self, **kwargs):
    obs = super(RandPlayer, self).reset(**kwargs)
    self.change_player = random() < 0.5
    if self.change_player:
      obs = list(obs)
      obs.reverse()
    return obs

  def step(self, actions):
    if self.change_player:
      actions = list(actions)
      actions.reverse()
    obs, rwd, done, info = self.env.step(actions)
    if self.change_player:
      obs = list(obs)
      obs.reverse()
      rwd = list(rwd)
      rwd.reverse()
    return obs, rwd, done, info


class VecRwdTransform(RewardWrapper):
  """ Reward Wrapper for sc2 full game """

  def __init__(self, env, weights):
    super(VecRwdTransform, self).__init__(env)
    self.weights = weights

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    rwd = [np.array(self.weights).dot(np.array(reward)) for reward in rwd]
    return obs, rwd, done, info


class StepMul(Wrapper):
  def __init__(self, env, step_mul=3 * 60 * 4):
    super(StepMul, self).__init__(env)
    self._step_mul = step_mul
    self._cur_obs = None

  def reset(self, **kwargs):
    self._cur_obs = self.env.reset()
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space
    return self._cur_obs

  def step(self, actions):
    done, info = False, {}
    cumrew = [0.0 for _ in actions]  # number players
    for _ in range(self._step_mul):
      self._cur_obs, rew, done, info = self.env.step(actions)
      cumrew = [a + b for a, b in zip(cumrew, rew)]
      if done:
        break
    return self._cur_obs, cumrew, done, info


class AllObs(ObservationWrapper):
    """ Give all players' observation to cheat_players
        cheat_players = None means all players are cheating,
        cheat_players = [] means no one is cheating) """

    def __init__(self, env, cheat_players=None):
        super(AllObs, self).__init__(env)
        self.observation_space = NoneSpace()
        self.cheat_players = cheat_players

    def observation(self, obs):
        observation = []
        for i in range(len(obs)):
            if i in self.cheat_players:
                if isinstance(obs[0], list) or isinstance(obs[0], tuple):
                    observation.append(list(chain(*obs[i:], *obs[0:i])))
                else:
                    observation.append(list(obs[i:]) + list(obs[0:i]))
            else:
                observation.append(obs[i])
        return observation

    def reset(self):
        obs = self.env.reset()
        self.action_space = self.env.action_space
        obs_space  = self.env.observation_space
        assert isinstance(obs_space, spaces.Tuple)
        assert all([sp == obs_space.spaces[0] for sp in obs_space.spaces])
        n_player = len(obs_space.spaces)
        if self.cheat_players is None:
            self.cheat_players = range(n_player)
        if isinstance(obs_space.spaces[0], spaces.Tuple):
            sp = spaces.Tuple(obs_space.spaces[0].spaces * n_player)
        else:
            sp = spaces.Tuple(obs_space.spaces[0] * n_player)
        sps = [sp if i in self.cheat_players else obs_space.spaces[i]
               for i in range(n_player)]
        self.observation_space = spaces.Tuple(sps)
        return self.observation(obs)


class OppoObsAsObs(Wrapper):
  """ A base wrapper for appending (part of) the opponent's obs to obs """

  def __init__(self, env):
    super(OppoObsAsObs, self).__init__(env)
    self._me_id = 0
    self._oppo_id = 1

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    return self._process_obs(obs)

  def _expand_obs_space(self, **kwargs):
    raise NotImplementedError("Implement your own func.")

  def _parse_oppo_obs(self, raw_oppo_obs):
    raise NotImplementedError("Implement your own func.")

  def _append_obs(self, self_obs, raw_oppo_obs):
    if isinstance(self_obs, tuple):
      return self_obs + self._parse_oppo_obs(raw_oppo_obs)
    elif isinstance(self_obs, dict):
      self_obs.update(self._parse_oppo_obs(raw_oppo_obs))
      return self_obs
    else:
      raise Exception("Unknown obs type in OppoObsAsObs wrapper.")

  def _process_obs(self, obs):
    if obs[0] is None:
      return obs
    else:
      appended_self_obs = self._append_obs(
        obs[self._me_id], self.env.unwrapped._obs[self._oppo_id])
      return [appended_self_obs, obs[self._oppo_id]]

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    assert len(obs) == 2, "OppoObsAsObs only supports 2 players game."
    return self._process_obs(obs), rwd, done, info


