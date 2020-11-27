""" Arena compatible vizdoom env """
import os
from copy import deepcopy
import random
from collections import deque, Counter
from portpicker import pick_unused_port

from numpy.core._multiarray_umath import ndarray
import vizdoom as vd
import numpy as np
import gym
from gym import Wrapper
from gym.spaces import Tuple, Box, MultiDiscrete

from arena.utils.vizdoom.run_parallel import RunParallel
from arena.utils.vizdoom.player import PlayerHostConfig, PlayerJoinConfig, PlayerConfig
from arena.utils.vizdoom.player import player_setup, player_host_setup, player_join_setup
from arena.utils.vizdoom.core_env import _get_screen_shape, _get_action_dim, _get_available_game_variables_dim
#from arena.wrappers.vizdoom.reward_shape import RwdShapeWu2, RwdShapeWu3
from arena.interfaces.vizdoom.act_int import Discrete6ActionInt


class PlayerEnv(gym.Env):
    """ViZDoom per player environment, tuned for CIG Track 1 Death Match.

    Use Wu Yuxing's trick to enhance the action.
    TODO(pengsun): move the action-enhancement logic to an Interface?"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.game = None

        self.observation_space = Box(low=0, high=255, dtype=np.uint8,
                    shape=_get_screen_shape(cfg.screen_resolution, cfg.screen_format))
        self.action_space = MultiDiscrete([2] * _get_action_dim(cfg))

        self._state = None
        self._obs = None
        self._rwd = None
        self._done = None
        self._act = None
        self._game_var_list = None
        self._game_vars = {}
        self._game_vars_pre = {}

        self.last_history = deque(maxlen=60)
        self.current_ammo = 15
        self.all_actions = [
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 0 move fast forward
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 fire
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 2 move left
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 3 move right
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # 4 turn left
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 5 turn right
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 20], # 6 turn left 20 degree and move forward
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 20], # 7 turn right 20 degree and move forward
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 8 move forward
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9 turn 180
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 10 move left
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 11 move right
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12 turn left
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 13 turn right
        ]

    def repeat_action(self, action):
        cnt = 1
        while len(self.last_history) > cnt + 1 \
                and self.last_history[-(cnt+1)] == action:
            cnt += 1
        return cnt

    def act(self, action):
        """Convert to Discrete(6) action to the full allowed action."""
        # TODO(pengsun): an Interface for the logic?
        self.last_history.append(action)
        is_attacking = (self.all_actions[1] in list(self.last_history)[-3:])
        # if is_attacking:
        #     self.cfg.repeat_frame = 3
        # else:
        #     self.cfg.repeat_frame = 3
        if action in [self.all_actions[4], self.all_actions[5]]:
            if self.repeat_action(action) > 3:
                # print(self.last_history)
                action = self.all_actions[6] if action == self.all_actions[4] else self.all_actions[7]
        if action in [self.all_actions[0], self.all_actions[2], self.all_actions[3]]:
            if self.repeat_action(action) % 16 == 0:
                action = self.all_actions[9]
                # print('repeat_action')
        if action in [self.all_actions[0], self.all_actions[2], self.all_actions[3], self.all_actions[4], self.all_actions[5]]:
            if is_attacking:
                if action == self.all_actions[0]:
                    action = self.all_actions[8]
                if action == self.all_actions[2]:
                    action = self.all_actions[10]
                if action == self.all_actions[3]:
                    action = self.all_actions[11]
                if action == self.all_actions[4]:
                    action = self.all_actions[12]
                if action == self.all_actions[5]:
                    action = self.all_actions[13]
        if self.game.is_player_dead():
            self.last_history.clear()
        return action

    def reset(self):
        if not self.cfg.is_multiplayer_game:
            if self.game is None:
                self._init_game()
            self.game.new_episode()
        else:
            self._init_game()
            if self.cfg.num_bots > 0:
                self._add_bot()

        self._state, self._obs, self._done = self._grab()
        self._update_vars()
        return self._obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        a = self.act(action)
        # if a!=action:
        #     print(a)
        self._rwd = self.game.make_action(a, self.cfg.repeat_frame)
        self._state, self._obs, self._done = self._grab()
        self._update_vars()

        return self._obs, self._rwd, self._done, {}

    def close(self):
        if self.game:
            self.game.close()

    def render(self, *args):
        return self._obs

    def _init_game(self):
        self.close()

        game = vd.DoomGame()
        game = player_setup(game, self.cfg)
        if self.cfg.is_multiplayer_game:
            if self.cfg.host_cfg is not None:
                game = player_host_setup(game, self.cfg.host_cfg)
            elif self.cfg.join_cfg is not None:
                game = player_join_setup(game, self.cfg.join_cfg)
            else:
                raise ValueError('neither host nor join, error!')
        game.init()
        self.game = game
        self._game_var_list = self.game.get_available_game_variables()
        self._update_vars()

    def _grab(self):
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        if done:
            obs = np.ndarray(shape=self.observation_space.shape,
                             dtype=self.observation_space.dtype)
        else:
            obs = state.screen_buffer
        return state, obs, done

    def _add_bot(self):
        self.game.send_game_command("removebots")
        for i in range(self.cfg.num_bots):
            self.game.send_game_command("addbot")

    def _update_vars(self):
        self._game_vars_pre = deepcopy(self._game_vars)
        if self.unwrapped._state is not None:  # ensure current frame is available
            for key in self._game_var_list:
                key_name = str(key)
                self._game_vars[key_name[key_name.find('.')+1:]] = self.game.get_game_variable(key)
            # Fix 'HEALTH == -999900.0' error when respawn
            if self._game_vars['HEALTH'] < 0.0:
                self._game_vars['HEALTH'] = 0.0


class VecEnv(gym.Env):
    """Vectorize a list of environments as a single environment."""
    def __init__(self, envs):
        self.observation_space = Tuple([e.observation_space for e in envs])
        self.action_space = Tuple([e.action_space for e in envs])

        self._envs = envs
        self._par = RunParallel()

    def reset(self):
        # self.close()
        # os.system('pkill -9 vizdoom')
        observations = self._par.run((e.reset) for e in self._envs)
        return observations

    def step(self, actions):
        ret = self._par.run((e.step, act)
                            for e, act in zip(self._envs, actions))
        observations, rewards, dones, infos = [item for item in zip(*ret)]
        # print('VecEnv/infos')
        # print(infos)
        # return observations, rewards, dones, infos
        return observations, rewards, dones, {}

    def close(self):
        self._par.run((e.close) for e in self._envs)

    def render(self, *args):
        obs = self._par.run((e.render) for e in self._envs)
        return obs

    @property
    def envs(self):
        return self._envs


class VizdoomMPEnv(gym.Env):
    """ViZDoom multi-player environment."""
    def __init__(self, config_path,
                 num_players=2,
                 num_bots=0,
                 mode='train',
                 max_steps=2100,
                 episode_timeout=2100,
                 is_window_visible=False,
                 is_window_cv_visible=False):

        self.port = pick_unused_port()
        self.mode = mode
        # host cfg
        self.host_cfg = PlayerHostConfig(self.port)
        self.host_cfg.num_players = num_players
        # join cfg
        self.join_cfg = PlayerJoinConfig(self.port)
        # player cfg
        self.players_cfg = []
        for i in range(self.host_cfg.num_players):
            cfg = PlayerConfig()
            cfg.config_path = config_path
            cfg.player_mode = vd.Mode.PLAYER
            cfg.screen_resolution = vd.ScreenResolution.RES_800X450
            cfg.screen_format = vd.ScreenFormat.CBCGCR
            cfg.is_window_visible = is_window_visible
            # cfg.ticrate = vd.DEFAULT_TICRATE * 2
            cfg.episode_timeout = episode_timeout
            if i == 0:  # host
                cfg.host_cfg = self.host_cfg
                cfg.name = 'WhoAmI'
                cfg.num_bots = num_bots
            else:
                cfg.join_cfg = self.join_cfg
                cfg.name = 'P{}'.format(i)
            self.players_cfg.append(cfg)

        # the player-wise env and the vec-env
        # TODO(jackzbzheng): add different observation wrappers
        self.envs = []
        for cfg in self.players_cfg:
            e = PlayerEnv(cfg)
            #e = RwdShapeWu2(e, dist_penalty_thres=3)
            self.envs.append(e)
        self.env = VecEnv(self.envs)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, actions):
        observations, rewards, dones, infos = self.env.step(actions)
        done = all(dones)
        # print('VizdoomMPEnv/infos1')
        # print(infos)
        # Will be reassigned when game done in reward_wrapper
        infos['outcome'] = [0 for _ in range(len(self.players_cfg))]
        infos['frag'] = [0 for _ in range(len(self.players_cfg))]
        infos['navigation'] = [0 for _ in range(len(self.players_cfg))]
        # infos['outcome'] = [0]
        # infos['frag'] = [0]
        # infos['navigation'] = [0]
        # print('VizdoomMPEnv/infos2')
        # print(infos)
        return observations, rewards, done, infos

    def reset(self):
        return self.env.reset()


class VizdoomVecRwd(Wrapper):
    """Vizdoom vectorized-reward wrapper, tuned to only the CIG track 1 map.

    Make a reward vector (instead of a conventional scalar) for EACH player.
    TODO(pengsun): make it a per-player wrapper."""
    def __init__(self, env):
        super(VizdoomVecRwd, self).__init__(env)
        self.envs = self.env.envs
        assert hasattr(env, 'players_cfg'), 'missing field players_cfg. Must wrap over an VizdoomMPEnv'
        self.num_players = len(env.players_cfg)
        assert self.num_players >= 2, 'at least 2 players in multi-agent/player case.'

    def reset(self, **kwargs):
        self._step = 0
        self.obs = self.env.reset(**kwargs)
        self.cumulated_frag = np.zeros(self.num_players)
        self.cumulated_navigate = np.zeros(self.num_players)
        # self.cumulated_rw = np.zeros(1)
        self.address_list = np.zeros((self.num_players, 18))  # TODO(pengsun): demystify the 18
        self.adress_saver = np.zeros(self.num_players)
        self.adress_sum = np.zeros(self.num_players)
        # self.cumulated_frag = np.zeros(1)
        # self.cumulated_navigate = np.zeros(1)
        # # self.cumulated_rw = np.zeros(1)
        # self.address_list = np.zeros((1, 14))
        # self.adress_saver = np.zeros(1)
        # self.adress_sum = np.zeros(1)
        return self.obs

    def rwd_transform(self, reward, game_var_pre, game_var, i):
        # TODO(pengsun): refactor below code
        # Dim: 7+1
        # major
        #death_diff = game_var['DEATHCOUNT'] - game_var_pre['DEATHCOUNT']
        #rwd_live = -3.0 if death_diff > 0.0 else 0.0

        rwd_kill_frag = 0
        rwd_killed_frag = 0
        rwd_attack = 0
        rwd_hit = 0
        rwd_hitt = 0
        rwd_badshot = 0
        rwd_armo = 0
        rwd_shot = 0
        rwd_heal = 0
        rwd_shot1 = 0
        rwd_shot2 = 0

        frag_diff = game_var['FRAGCOUNT'] - game_var_pre['FRAGCOUNT']
        if frag_diff > 0:
            rwd_kill_frag = 1
        if frag_diff < 0:
            rwd_killed_frag = -1

        # TODO(pengsun): unavailable in vd 1.7.1?
        rwd_hit = 1.0 if (game_var['HITCOUNT'] - game_var_pre['HITCOUNT']) > 0.0 else 0.0
        # TODO(pengsun): unavailable in vd 1.7.1?
        hits_taken_diff = game_var['HITS_TAKEN'] - game_var_pre['HITS_TAKEN']
        rwd_hitt = 0.0 if hits_taken_diff == 0.0 else -1.0

        health_diff = game_var['HEALTH'] - game_var_pre['HEALTH']
        if 5 < health_diff <= 25:
            rwd_heal = 1
        elif health_diff < 0:
            rwd_attack = -1
        ammo_diff = game_var['SELECTED_WEAPON_AMMO'] - game_var_pre['SELECTED_WEAPON_AMMO']

        if ammo_diff == 1 or ammo_diff == 2 or ammo_diff == 5:
            rwd_shot1 = 1
        if ammo_diff == -1 and rwd_hit == 0:
            rwd_shot2 = -1

        #kill_diff = game_var['KILLCOUNT'] - game_var_pre['KILLCOUNT']
        #rwd_kill = 100.0 if kill_diff > 0.0 else 0.0

        armo_diff = game_var['ARMOR'] - game_var_pre['ARMOR']
        if armo_diff > 0.0:
            rwd_armo = 1.0
        move_diff = self.cal_move_distance(game_var['POSITION_X'], game_var['POSITION_Y'],
                                           game_var_pre['POSITION_X'], game_var_pre['POSITION_Y'])
        # if 1165 < game_var['POSITION_Y'] < 1296 and \
        #     (-432 < game_var['POSITION_X'] < 80 or 944 < game_var['POSITION_X'] < 1360) and \
        #     move_diff > 3 and health_diff >= 0:
        #     rwd_move = 0.003
        # elif 144 < game_var['POSITION_X'] < 880 and 144 < game_var['POSITION_Y'] < 880 and move_diff > 4.0:
        #     rwd_move = -0.001
        # elif 144 < game_var['POSITION_X'] < 880 and 144 < game_var['POSITION_Y'] < 880 and move_diff < 4.0:
        #     rwd_move = -0.002
        # elif -432 < game_var['POSITION_X'] < 80 and game_var['POSITION_Y'] < 700:
        #     rwd_move = -0.001
        # elif 944 < game_var['POSITION_X'] < 1360 and game_var['POSITION_Y'] < 200:
        #     rwd_move = -0.001
        # elif 80 < game_var['POSITION_X'] < 944 and game_var['POSITION_Y'] < 85:
        #     if 450 < game_var['POSITION_X'] < 580 and game_var['POSITION_Y'] > -16:
        #         rwd_move = -0.002
        #     else:
        #         rwd_move = -0.001
        # elif move_diff > 4.0 and health_diff >= 0:
        #     rwd_move = 0.001
        # elif move_diff < 0.6:
        #     rwd_move = -0.006
        # else:
        #     rwd_move = -0.0005


        if move_diff > 4.0 and health_diff >= 0:
            rwd_move = 0.001
        elif move_diff < 0.6:
            rwd_move = -0.006
        else:
            rwd_move = -0.0005

        binary_address = self.judge_address(game_var['POSITION_X'], game_var['POSITION_Y'], i)
        self.adress_sum[i] = np.sum(binary_address[i], axis=0)

        death_diff = game_var['DEATHCOUNT'] - game_var_pre['DEATHCOUNT']
        if death_diff != 0 or self.adress_sum[i] == 18:
            self.address_list[i] = np.zeros(18)
            binary_address[i] = np.zeros(18)
            self.adress_sum[i] = 0
            self.adress_saver[i] = 0
        elif self.adress_saver[i] != self.adress_sum[i]:
            self.adress_saver[i] = self.adress_sum[i]
            rwd_move = 1
            # rwd_move = 0.05
        # print('XXXXXXXXXXXXX')
        # print(game_var['POSITION_X'])
        # print('YYYYYYYYYYYYY')
        # print(game_var['POSITION_Y'])
        # if rwd_move ==0.05:
        # print('rwd_move')
        # print(rwd_move)
        # if self.adress_sum[i] != 0:
        #     print('adress_sum')
        #     print(i)
        #     print(self.adress_sum)
        #     print('binary_address')
        #     print(binary_address)

        # print('DEATH  : {} -> {} = {}'.format(game_var_pre['DEATHCOUNT'], game_var['DEATHCOUNT'], rwd_live))
        # print('FRAG   : {} -> {} = {}'.format(game_var_pre['FRAGCOUNT'], game_var['FRAGCOUNT'], rwd_frag))
        # print('HIT    : {} -> {} = {}'.format(game_var_pre['HITCOUNT'], game_var['HITCOUNT'], rwd_hit))
        # print('HITT   : {} -> {} = {}'.format(game_var_pre['HITS_TAKEN'], game_var['HITS_TAKEN'], rwd_hitt))
        # print('HEALTH : {} -> {} = {}'.format(game_var_pre['HEALTH'], game_var['HEALTH'], rwd_heal))
        # print('AMMO   : {} -> {} = {}'.format(game_var_pre['SELECTED_WEAPON_AMMO'], game_var['SELECTED_WEAPON_AMMO'], rwd_shot))
        # print('KILL   : {} -> {} = {}'.format(game_var_pre['KILLCOUNT'], game_var['KILLCOUNT'], rwd_kill))
        # print('ARMOR  : {} -> {} = {}'.format(game_var_pre['ARMOR'], game_var['ARMOR'], rwd_armo))
        # print('MOVE   : {} = {}'.format(move_diff, rwd_move))

        # minor
        # rwd_pick = game_var['ITEMCOUNT'] - game_var_pre['ITEMCOUNT']
        # if rwd_pick > 0:
        #     print('PICK: {} -> {} = {}'.format(game_var_pre['ITEMCOUNT'], game_var['ITEMCOUNT'], rwd_pick))
        # rwd_find = game_var['SECRETCOUNT'] - game_var_pre['SECRETCOUNT']
        # if rwd_find > 0:
        #     print('FIND: {} -> {} = {}'.format(game_var_pre['SECRETCOUNT'], game_var['SECRETCOUNT'], rwd_find))

        return (reward, rwd_kill_frag, rwd_killed_frag, rwd_hit, rwd_hitt, rwd_heal, rwd_shot1, rwd_shot2, rwd_armo, rwd_move)

    def step(self, action):
        # print('Steps: {}'.format(self._step))
        self._step += 1
        obs, rwds, done, info = self.env.step(action)

        rwds = [
            self.rwd_transform(rwds[i],
                               self.envs[i].unwrapped._game_vars_pre,
                               self.envs[i].unwrapped._game_vars,
                               i)
            for i in range(self.num_players)
        ]
        self.cumulated_frag += np.array([
            np.sum(rwd_p[1:3]) for rwd_p in rwds  # see rwd_transform for the index 1:3
        ])
        self.cumulated_navigate += np.array([
            np.sum(rwd_p[9]) for rwd_p in rwds  # see rwd_transform for the index 9
        ])

        if done:
            FRAG_arr = []
            Navigate_arr = []
            for i in range(self.num_players):
                FRAG_arr = np.append(FRAG_arr, [self.cumulated_frag[i]])
                Navigate_arr = np.append(Navigate_arr, self.cumulated_navigate[i])
            FRAG_sort_index = np.argsort(-FRAG_arr)

            # Decide the outcome in [+1, 0, -1] according to the rank
            # For example, when num_players = 8, the below logic is equivalent to:
            #   outcome[FRAG_sort_index[0]] = 1
            #   outcome[FRAG_sort_index[1]] = 1
            #   outcome[FRAG_sort_index[2]] = 0
            #   outcome[FRAG_sort_index[3]] = 0
            #   outcome[FRAG_sort_index[4]] = 0
            #   outcome[FRAG_sort_index[5]] = -1
            #   outcome[FRAG_sort_index[6]] = -1
            #   outcome[FRAG_sort_index[7]] = -1
            # when num_players = 2, there is no tie and the below logic is equivalent to:
            #   outcome[FRAG_sort_index[0]] = 1
            #   outcome[FRAG_sort_index[1]] = -1
            outcome: ndarray = np.zeros(self.num_players)
            assert self.num_players >= 2
            if self.num_players == 2:
                ind_win = ind_lose = 1  # there is no tie
            else:  # >= 3
                ind_win = int(float(self.num_players) * (1/3.0))
                ind_lose = int(float(self.num_players) * (2/3.0))
            for i in range(self.num_players):
                if i < ind_win:
                    # ranked top, deemed as win
                    outcome[FRAG_sort_index[i]] = 1
                elif ind_win <= i < ind_lose:
                    # ranked middle, deemed as tie
                    outcome[FRAG_sort_index[i]] = 0
                else:  # ind_lose <= i
                    # ranked bottom, deemed as lose
                    outcome[FRAG_sort_index[i]] = -1

            info['outcome'] = outcome
            # info['debug'] = [1.0, 2.3, 43, 23.5, 412.4, 413, 41, 0.532]
            info['frag'] = FRAG_arr
            info['navigation'] = Navigate_arr
            # Replace the original rwd_p[0] with outcome
            rwds = [(outcome[i],) + rwd_p[1:] for i, rwd_p in enumerate(rwds)]
            # print('--------[Done]--------')
        # sum = 0
        # for i in range(1, len(rwds[0])):
        #    sum += rwds[0][i]
        # if sum != -0.001 and sum != -0.005 and sum != -0.002 and sum != 0.001:
        #     print(sum)
        #     print(rwds)
        #     print(info)
        return obs, rwds, done, info

    def cal_move_distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return np.sqrt(dx * dx + dy * dy)

    def judge_address(self, x, y, i):
        # address_list = np.zeros(12)
        if -432 < x < -304:
            if 1162 < y < 1296:
                self.address_list[i][0] = 1
            elif 784 < y < 885:
                self.address_list[i][1] = 1
        elif -304 < x < -16:
            if 400 < y < 600:
                self.address_list[i][2] = 1
        elif -16 < x < 80:
            # if 940 < y < 1008:
            #     self.address_list[i][3] = 1
            if 1165 < y < 1296:
                self.address_list[i][3] = 1
            elif -16 < y < 84:
                self.address_list[i][4] = 1
        elif 144 < x < 272:
            if 780 < y < 880:
                self.address_list[i][5] = 1
            elif 144 < y < 250:
                self.address_list[i][6] = 1
        elif 396 < x < 432 and 1423 < y < 1488:
            self.address_list[i][7] = 1
        elif 412 < x < 610:
        #     if 1070 < y < 1230:
        #         self.address_list[i][8] = 1
            if 780 < y < 880:
                self.address_list[i][8] = 1
        # elif 350 < y < 540:
        #     self.address_list[i][10] = 1
        # elif -16 < y < 84:
        #     self.address_list[i][11] = 1
        elif 847 < x < 880 and 1423 < y < 1488:
            self.address_list[i][9] = 1
        elif 734 < x < 880:
            if 780 < y < 880:
                self.address_list[i][10] = 1
            elif 144 < y < 250:
                self.address_list[i][11] = 1
        elif 700 < x < 784 and -336 < y < -230:
            self.address_list[i][12] = 1
        elif 944 < x < 1040:
            # if 940 < y < 1008:
            #     self.address_list[i][7] = 1
            if 1165 < y < 1296:
                self.address_list[i][13] = 1
            # elif 447 < y < 573:
            #     self.address_list[i][8] = 1
            elif -16 < y < 84:
                self.address_list[i][14] = 1
        elif 1230 < x < 1360:
            if 1165 < y < 1296:
                self.address_list[i][15] = 1
            elif 700 < y < 816:
                self.address_list[i][16] = 1
            elif 208 < y < 320:
                self.address_list[i][17] = 1
        return self.address_list


def main():
    from arena.env.env_int_wrapper import EnvIntWrapper
    from arena.interfaces.raw_int import RawInt
    from arena.interfaces.vizdoom.obs_int import FrameVarObsInt, ReshapedFrameObsInt

    num_players = 3
    num_bots = 2
    env = VizdoomMPEnv(config_path='./../utils/vizdoom/_scenarios/cig.cfg',
                       num_players=num_players,
                       num_bots=num_bots,
                       episode_timeout=50000,
                       is_window_visible=True)
    env = VizdoomVecRwd(env)

    def _install_interfaces(i_agent):
        inter = RawInt()
        inter = ReshapedFrameObsInt(inter, env.envs[i_agent])
        inter = Discrete6ActionInt(inter)
        return inter

    env = EnvIntWrapper(env, [_install_interfaces(i) for i in range(num_players)])
    for episode in range(10):
        obs = env.reset()
        ep_return = None
        ep_step = 0
        # print('episode {}'.format(episode))
        while True:
            act = env.action_space.sample()
            # act = 0
            obs, rwd, done, info = env.step(act)
            #if ep_return is None:
                # ep_return = rwd
             #   ep_return = (rwd[0][0], rwd[1][0])
            #else:
                # ep_return = [a+b for a, b in zip(ep_return, rwd)]
             #   ep_return = [a+b for a, b in zip(ep_return, (rwd[0][0], rwd[1][0]))]
            if done:
                print('ep steps: {}; ep return: {}'.format(ep_step, ep_return))
                break
            else:
                ep_step += 1

if __name__ == '__main__':
    main()
