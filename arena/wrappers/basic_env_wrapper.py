from gym import ObservationWrapper
from gym import Wrapper, RewardWrapper
from gym import spaces
from itertools import chain
from arena.utils.unit_util import merge_units
from arena.utils.unit_util import collect_units_by_alliance, collect_units_by_types
from arena.utils.constant import AllianceType, UNIT_TYPEID
from arena.utils.spaces import NoneSpace
from random import random
import numpy as np
from timitate.utils.commands import cmd_camera, cmd_with_pos, cmd_with_tar, noop


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


class EpisodicLife(Wrapper):
  """ Make end-of-life == end-of-episode, but only reset on true game over. """
  def __init__(self, env, max_lives=100, max_step=500):
    super(EpisodicLife, self).__init__(env)
    self.max_lives = max_lives
    self.lives = 0
    self.steps = 0
    self.max_step = max_step
    self.was_real_done = True

  def check_done(self, obs):
    dones = []
    if len(obs) == 2:
      for timestep in obs:
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
        dones.append(len(my_units) == 0)
    else:
      timestep = obs[0]
      units = timestep.observation.raw_data.units
      my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
      enemy_units = collect_units_by_alliance(units, AllianceType.ENEMY.value)
      lose = len(my_units) == 0
      win = len(enemy_units) == 0
      dones.append(lose)
      dones.append(win)
    return dones

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.was_real_done = done
    self.steps += 1
    if self.was_real_done:
      return obs, reward, done, info
    # check current units
    dones = self.check_done(obs)
    if len(obs) == 2:
      done = any(dones)
      if done:
        reward = [-1 if d else 1 for d in dones]
      if self.steps >= self.max_step:
        done = True
        reward = [-1 for d in dones]
        self.was_real_done = True
    else:
      lose = dones[0]
      win = dones[1]
      if lose:
        done = True
        reward = [-1]
      elif win:
        done = True
        reward = [1]
      if self.steps >= self.max_step:
        done = True
        reward = [-1]
        self.was_real_done = True

    return obs, reward, done, info

  def reset(self, **kwargs):
    """ Reset only when lives are exhausted. """
    self.steps = 0
    if self.was_real_done or self.lives >= self.max_lives:
      obs = self.env.reset(**kwargs)
      self.lives = 0
    else:
      while True:
        obs, _, _, _ = self.env.step([[]] * len(self.env.action_space.spaces))
        if not any(self.check_done(obs)):
          self.lives += 1
          break
    return obs


class VecRwd(RewardWrapper):
  """ Reward Wrapper for sc2 full game """

  def __init__(self, env, append=False, scale=(1.0 / 1000, 1.0 / 100, 1.0 / 1000)):
    super(VecRwd, self).__init__(env)
    self.last_scores = []
    self.append = append
    self.gas_mineral_ratio = 1.25  # roughly estimate: 5 mineral per worker / 4 gas per worker
    self.scale = scale

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self.last_scores = [o.observation.score.score_details for o in self.unwrapped._obs]
    return obs

  @staticmethod
  def category_sum(score_details):
    return score_details.none + score_details.army + score_details.economy + \
           score_details.technology + score_details.upgrade

  @staticmethod
  def vital_sum(score_details):
    return score_details.life + score_details.shields + score_details.enegy

  def compute_rwd(self, last_score, score):
    # prod
    used_m_diff = self.category_sum(score.total_used_minerals) - \
                  self.category_sum(last_score.total_used_minerals)
    used_v_diff = self.category_sum(score.total_used_vespene) - \
                  self.category_sum(last_score.total_used_vespene)
    rwd_prod = used_m_diff + self.gas_mineral_ratio * used_v_diff

    # resource
    collect_m_diff = score.collection_rate_minerals - \
                     last_score.collection_rate_minerals
    collect_v_diff = score.collection_rate_vespene - \
                     last_score.collection_rate_vespene
    rwd_res = collect_m_diff + self.gas_mineral_ratio * collect_v_diff

    # combat
    kill_m_diff = self.category_sum(score.killed_minerals) - \
                  self.category_sum(last_score.killed_minerals)
    kill_v_diff = self.category_sum(score.killed_vespene) - \
                  self.category_sum(last_score.killed_vespene)
    loss_m_diff = self.category_sum(score.lost_minerals) - \
                  self.category_sum(last_score.lost_minerals)
    loss_v_diff = self.category_sum(score.lost_vespene) - \
                  self.category_sum(last_score.lost_vespene)
    kill_value = kill_m_diff + self.gas_mineral_ratio * kill_v_diff
    lost_value = loss_m_diff + self.gas_mineral_ratio * loss_v_diff
    rwd_combat = kill_value - lost_value

    rwd = [rwd_prod, rwd_res, rwd_combat]
    return [rwd * scale for rwd, scale in zip(rwd, self.scale)]

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    scores = [o.observation.score.score_details for o in self.unwrapped._obs]
    rwd_new = [self.compute_rwd(last_score, score)
               for last_score, score in zip(self.last_scores, scores)]
    self.last_scores = scores
    if self.append:
      rwd = [[r] + r_new if type(r) is not list else r + r_new
             for r, r_new in zip(rwd, rwd_new)]
    else:
      rwd = rwd_new
    return obs, np.array(rwd), done, info


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


class NoResetDenseRWD(Wrapper):
    def __init__(self, env, max_lives=100, max_step=500):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        """
        super(NoResetDenseRWD, self).__init__(env)
        self.max_lives = max_lives
        self.lives = 0
        self.steps = 0
        self.max_step = max_step
        self.was_real_done = True

    def check_done(self, obs):
        dones = []
        if len(obs) == 2:
            for timestep in obs:
                units = timestep.observation.raw_data.units
                my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
                dones.append(len(my_units)==0)
        else:
            timestep = obs[0]
            units = timestep.observation.raw_data.units
            my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
            enemy_units = collect_units_by_alliance(units, AllianceType.ENEMY.value)
            lose = len(my_units)==0
            win = len(enemy_units)==0
            dones.append(lose)
            dones.append(win)
        return dones

    def compute_dense_reward(self, last_obs, obs):
        if len(obs) == 2:
            return 0
        else:
            last_timestep = last_obs[0]
            timestep = obs[0]
            last_units = last_timestep.observation.raw_data.units
            units = timestep.observation.raw_data.units
            last_my_units = collect_units_by_alliance(last_units, AllianceType.SELF.value)
            my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
            last_enemy_units = collect_units_by_alliance(last_units, AllianceType.ENEMY.value)
            enemy_units = collect_units_by_alliance(units, AllianceType.ENEMY.value)

            rwd = 0
            rwd += len(my_units) - len(last_my_units)
            rwd += len(last_enemy_units) - len(enemy_units)
            rwd *= 0.1
            return rwd

    def step(self, action):
        last_obs = self.env._obs
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        self.steps += 1
        if self.was_real_done:
            return obs, reward, done, info
        # check current units
        dones = self.check_done(obs)
        if len(obs) == 2:
            done = any(dones)
            if done:
                reward = [-1 if d else 1 for d in dones]
            elif self.steps >= self.max_step:
                done = True
                reward = [-1 for d in dones]
                self.was_real_done = True
            else:
                rwd_0 = self.compute_dense_reward([last_obs[0]], [obs[0]])
                rwd_1 = -rwd_0
                reward = [rwd_0, rwd_1]
        else:
            lose = dones[0]
            win = dones[1]
            if lose:
                done = True
                reward = [-1]
            elif win:
                done = True
                reward = [1]
            elif self.steps >= self.max_step:
                done = True
                reward = [-1]
                self.was_real_done = True
            else:
                reward = self.compute_dense_reward(last_obs, obs)
                reward = [reward]
            # if reward[0] != 0:
            #     print(reward)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """ Reset only when lives are exhausted. """
        self.steps = 0
        if self.was_real_done or self.lives >= self.max_lives:
            obs = self.env.reset(**kwargs)
            self.lives = 0
        else:
            while True:
                obs, _, _, _ = self.env.step([[]]*len(self.env.action_space.spaces))
                if not any(self.check_done(obs)):
                    self.lives += 1
                    break
        return obs

    @property
    def obs_spec(self):
        return self.observation_space

    @property
    def action_spec(self):
        return self.action_space


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


class EarlyTerminate(RewardWrapper):
  """ Terminate the game early """

  def __init__(self, env):
    super(EarlyTerminate, self).__init__(env)

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    return obs

  def check_done(self, obs):
    dones = []
    for timestep in obs:
      scores = timestep.observation.score.score_details
      # print('mineral: ', scores.collection_rate_minerals)
      # print('vespene: ', scores.collection_rate_vespene)
      if scores.collection_rate_minerals < 1.0 and \
          scores.collection_rate_vespene < 1.0:
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, AllianceType.SELF.value)
        my_bases = collect_units_by_types(my_units,
                                          [UNIT_TYPEID.ZERG_HATCHERY.value,
                                           UNIT_TYPEID.ZERG_LAIR.value,
                                           UNIT_TYPEID.ZERG_HIVE.value])
        my_larvas = collect_units_by_types(my_units,
                                           [UNIT_TYPEID.ZERG_LARVA.value])
        my_drones = collect_units_by_types(my_units,
                                           [UNIT_TYPEID.ZERG_DRONE.value,
                                            UNIT_TYPEID.ZERG_DRONEBURROWED.value])
        my_minerals = timestep.observation.player_common.minerals
        if len(my_bases) == 0 and len(my_drones) == 0 and len(my_larvas) == 0:
          dones.append(True)
        elif len(my_bases) == 0 and my_minerals < 300:
          dones.append(True)
        elif len(my_bases) > 0 and len(my_drones) == 0 and my_minerals < 50:
          dones.append(True)
        else:
          dones.append(False)
      else:
        dones.append(False)
    return dones

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    if len(obs) == 2:
      early_dones = self.check_done(obs)
      if early_dones[0]:
        info['outcome'] = [-1.0, 1.0]
        return obs, [-1.0, 1.0], True, info
      if early_dones[1]:
        info['outcome'] = [1.0, -1.0]
        return obs, [1.0, -1.0], True, info
    else:
      print('WARN: EarlyTerminate wrapper only works for 2 players game.')
      pass
    return obs, rwd, done, info


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


class OppoTRTNoOut(Wrapper):
  """ Tower rush trick (TRT) wrapper, only for KairosJunction;
  5% probability to trigger """

  def __init__(self, env):
    super(OppoTRTNoOut, self).__init__(env)
    self._n_drones_trt = 2
    self._base_pos = [(31.5-7, 140.5+7), (120.5+7, 27.5-7)]
    self._good_spine_pos = [
      [(25.5, 147.5), (22, 143.5), (22, 140), (28.5, 149.5), (32.5, 150.5)],
      [(126, 20.5), (122, 18), (119, 17.5), (129.5, 24.5), (130, 28)]]
    # (37, 145), (36.5, 140.5), (35.5, 137), (29, 135.5), (27.5, 138)
    # (124, 32), (126.5, 34.5), (117, 31.5), (113.5, 25), (113.5, 21)
    self._me_id = 0
    self._oppo_id = 1

    self.is_abandon = False
    self.use_trt = False
    self._started_print = False
    self._completed_print = False
    self._executors = []

  def reset(self, **kwargs):
    if np.random.random() < 0.05:
      self.use_trt = True
      self.is_abandon = True
      print('Using tower rush trick.')
    else:
      self.use_trt = False
      self.is_abandon = False
    self._started_print = False
    self._completed_print = False
    self._executors = []

    obs = self.env.reset(**kwargs)
    return obs

  def step(self, actions):
    actions = self.act_trans(actions)
    obs, rwd, done, info = self.env.step(actions)
    if self.is_abandon:
      info['outcome'] = [None, None]
    return obs, rwd, done, info

  def _dist(self, x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

  def _ready_to_go(self, units):
    drones = [u for u in units if u.alliance == 1 and
              u.unit_type == UNIT_TYPEID.ZERG_DRONE.value]
    if len(self._executors) >= self._n_drones_trt:
      # update executors' attributes
      for i, e in enumerate(self._executors):
        if e is None:
          continue
        is_alive = False
        for d in drones:
          if d.tag == e.tag:
            self._executors[i] = d
            is_alive = True
            break
        if not is_alive:
          self._executors[i] = None
      return True
    for u in units:
      # once self spawning pool is on building after 0.3 progresses
      if u.alliance == 1 and u.unit_type == UNIT_TYPEID.ZERG_SPAWNINGPOOL.value \
        and u.build_progress > 0.05 and len(drones) > 0 \
        and len(self._executors) < self._n_drones_trt:
        for d in drones:
          if d not in self._executors:
            self._executors.append(d)
            if len(self._executors) >= self._n_drones_trt:
              return True
    return False

  def _mission_completed(self):
    # if there had been executors and now the executors have been eliminated (failed or success)
    if len(self._executors) == 0:
      return False
    for e in self._executors:
      if e is not None:
        return False
    return True

  def _target_pos(self, units):
    self_h = [u for u in units if u.alliance == 1 and
              u.unit_type == UNIT_TYPEID.ZERG_HATCHERY.value]
    if len(self_h) == 0:
      return 0, 0
    self_h0 = None
    for h in self_h:
      if min(self._dist(h.pos.x, h.pos.y,
                        self._base_pos[0][0], self._base_pos[0][1]),
             self._dist(h.pos.x, h.pos.y,
                        self._base_pos[1][0], self._base_pos[1][1])) < 10.0:  # see base_pos
        self_h0 = h
    if self_h0 is None:
      print('WARN: can not find self base, '
            'probably not KJ map in TRTActInt or be rushed.'
            'Tower rush trick closed.')
      self.use_trt = False
      return None, None
    if self._dist(self_h0.pos.x, self_h0.pos.y,
                  self._base_pos[0][0], self._base_pos[0][1]) < \
      self._dist(self_h0.pos.x, self_h0.pos.y,
                 self._base_pos[1][0], self._base_pos[1][1]):
      return self._base_pos[1], self._good_spine_pos[1]
    else:
      return self._base_pos[0], self._good_spine_pos[0]

  def _drone_micro(self, units):
    tar_pos, good_spine_pos = self._target_pos(units)
    if tar_pos is None or good_spine_pos is None:
      return []
    def _build_spinecrawler(drone):
      if len(drone.orders) > 0:
        if drone.orders[0].ability_id in [1166]:
          # Note: when fog covers the build pos, 1166 will not
          # be in orders, and instead it's move
          return noop()
        elif drone.orders[0].ability_id in [1, 16] and \
            (drone.orders[0].target_world_space_pos.x,
             drone.orders[0].target_world_space_pos.y) in good_spine_pos:
          return noop()
      build_pos = good_spine_pos[np.random.randint(0, len(good_spine_pos))]
      return cmd_with_pos(ability_id=1166,
                          x=build_pos[0],
                          y=build_pos[1],
                          tags=[drone.tag],
                          shift=False)

    def _move_to_tar(drone, tar_pos):
      if len(drone.orders) > 0:
        if drone.orders[0].ability_id in [1, 16]:
          if self._dist(drone.orders[0].target_world_space_pos.x,
                        drone.orders[0].target_world_space_pos.y,
                        tar_pos[0], tar_pos[1]) < 1.0:
            return noop()
      return cmd_with_pos(ability_id=1,
                          x=tar_pos[0],
                          y=tar_pos[1],
                          tags=[drone.tag],
                          shift=False)

    def _atk_tar(drone, tar_tag):
      if len(drone.orders) > 0:
        if drone.orders[0].ability_id in [23]:
          return noop()
      return cmd_with_tar(ability_id=23,
                          target_tag=tar_tag,
                          tags=[drone.tag],
                          shift=False)

    enemy_d = [u for u in units if u.alliance == 4 and
               u.unit_type == UNIT_TYPEID.ZERG_DRONE.value]
    pb_actions = []
    for i, drone in enumerate(self._executors):
      if drone is None:
        continue
      if i % 2 == 0:
        if self._dist(drone.pos.x, drone.pos.y, tar_pos[0], tar_pos[1]) < 5.0:
          pb_actions.append(_build_spinecrawler(drone))
        else:
          pb_actions.append(_move_to_tar(drone, tar_pos))
      else:
        if len(enemy_d) > 0:
          # enemy drones in drone's detect range
          enemy_d_tags = [u.tag for u in enemy_d]
          pb_actions.append(_atk_tar(drone, min(enemy_d_tags)))
        else:
          pb_actions.append(_move_to_tar(drone, tar_pos))
    return pb_actions

  def _trt_act(self, units):
    if self._ready_to_go(units):
      if not self._started_print:
        print('Launch tower rush trick.')
        self._started_print = True
      if not self._mission_completed():
        return self._drone_micro(units)
      else:
        if not self._completed_print:
          print('Tower rush trick completed.')
          self._completed_print = True
    return []

  def _get_cmd_tags(self, raw_acts):
    all_tags = []
    for a in raw_acts:
      if hasattr(a, 'action_raw') and hasattr(a.action_raw, 'unit_command') \
        and hasattr(a.action_raw.unit_command, 'unit_tags'):
        all_tags += a.action_raw.unit_command.unit_tags
    return all_tags

  def _remove_tags(self, ori_act, tags):
    if len(tags) == 0:
      return ori_act
    for a in ori_act:
      if hasattr(a, 'action_raw') and hasattr(a.action_raw, 'unit_command') \
        and hasattr(a.action_raw.unit_command, 'unit_tags'):
        a_tags = set(a.action_raw.unit_command.unit_tags)
        for t in tags:
          if t in a_tags:
            a_tags.remove(t)
        # protobuff repeated only support pop()
        while len(a.action_raw.unit_command.unit_tags) > 0:
          a.action_raw.unit_command.unit_tags.pop()
        for t in a_tags:
          a.action_raw.unit_command.unit_tags.append(t)
    return ori_act

  def act_trans(self, act):
    ori_act = act[self._oppo_id]
    if self.use_trt and \
        self.env.unwrapped._obs[self._oppo_id].observation.game_loop > 22.4*60*5:
      if not self._completed_print:
        print('Timeout. Tower rush trick closed.')
      self.use_trt = False
    if self.use_trt:
      units = self.env.unwrapped._obs[self._oppo_id].observation.raw_data.units
      trt_act = self._trt_act(units)
      trt_a_tags = self._get_cmd_tags(trt_act)
      ori_act = self._remove_tags(ori_act, trt_a_tags)
      # Note: the order of the added items matters;
      # if ori_act is placed before trt_act, the remained selection is determined
      # by trt_act and then the model will be confused.
      act[self._oppo_id] = trt_act + ori_act
    return act
