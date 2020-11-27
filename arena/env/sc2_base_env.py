""" gym compatible sc2 env """
import random
from gym import Env
from gym import spaces
from pysc2.env import sc2_env
from pysc2.env.sc2_env import SC2Env
from arena.utils.spaces import SC2RawObsSpace, SC2RawActSpace


class SC2BaseEnv(Env):
    def __init__(self, map_name="4MarineA",
                 players=(sc2_env.Agent(sc2_env.Race.zerg),
                          sc2_env.Bot(sc2_env.Race.zerg,
                                      sc2_env.Difficulty.very_hard)),
                 agent_interface_format=None,
                 agent_interface="feature",
                 max_steps_per_episode=10000,
                 screen_resolution=64,
                 screen_ratio=1.33,
                 camera_width_world_units=24,
                 minimap_resolution=64,
                 step_mul=4,
                 score_index=-1,
                 score_multiplier=1.0/1000,
                 disable_fog=False,
                 random_seed=None,
                 visualize=False,
                 max_reset_num=100,
                 save_replay_episodes=0,
                 replay_dir=None,
                 version=None,
                 update_game_info=False,
                 use_pysc2_feature=True,
                 game_core_config={}
                 ):
        self._version = version
        self.replay_dir = replay_dir
        self.save_replay_episodes = save_replay_episodes
        self.map_name = map_name
        self.players = players
        assert len(self.players) == 2
        assert all([isinstance(p, sc2_env.Agent) or
                    isinstance(p, sc2_env.Bot) for p in self.players])
        self.agent_players = [p for p in self.players
                              if isinstance(p, sc2_env.Agent)]
        self.max_steps_per_episode = max_steps_per_episode
        self.step_mul = step_mul
        self.disable_fog = disable_fog
        self.visualize = visualize
        self.random_seed = random_seed
        self.score_index = score_index
        self.score_multiplier = score_multiplier
        self.agent_interface_format = agent_interface_format
        self.use_pysc2_feature = use_pysc2_feature
        self.game_core_config = game_core_config
        self.update_game_info = update_game_info
        if agent_interface_format is None:
            self.agent_interface = agent_interface
            screen_res = (int(screen_ratio * screen_resolution) // 4 * 4,
                          screen_resolution)
            if agent_interface == 'rgb':
                self.agent_interface_format = \
                    sc2_env.AgentInterfaceFormat(
                        rgb_dimensions=sc2_env.Dimensions(
                            screen=screen_res,
                            minimap=minimap_resolution),
                        camera_width_world_units=camera_width_world_units)
            elif agent_interface == 'feature':
                self.agent_interface_format = \
                    sc2_env.AgentInterfaceFormat(
                        feature_dimensions=sc2_env.Dimensions(
                            screen=screen_res,
                            minimap=minimap_resolution),
                        camera_width_world_units=camera_width_world_units)

        self._reset_num = 0
        self.max_reset_num = max_reset_num
        self._start_env()
        self._gameinfo = self.env._controllers[0].game_info()

        self._obs = None
        self._rew = None
        self._done = None
        self._info = None

        self.observation_space = spaces.Tuple([SC2RawObsSpace()] * len(self.agent_players))
        self.action_space = spaces.Tuple([SC2RawActSpace()] * len(self.agent_players))

    def _start_env(self):
        if isinstance(self.map_name, list) or isinstance(self.map_name, tuple):
            map_name = random.choice(self.map_name)
            self.max_reset_num = 0
        else:
            map_name = self.map_name
        self.env = SC2Env(map_name=map_name,
                          players=self.players,
                          step_mul=self.step_mul,
                          agent_interface_format=self.agent_interface_format,
                          game_steps_per_episode=self.max_steps_per_episode,
                          disable_fog=self.disable_fog,
                          visualize=self.visualize,
                          random_seed=self.random_seed,
                          score_index=self.score_index,
                          score_multiplier=self.score_multiplier,
                          save_replay_episodes=self.save_replay_episodes,
                          replay_dir=self.replay_dir,
                          version=self._version,
                          use_pysc2_feature=self.use_pysc2_feature,
                          update_game_info=self.update_game_info,
                          **self.game_core_config,
                          )

    def reset(self, **kwargs):
        self._reset_num += 1
        if self._reset_num > self.max_reset_num >=0:
            self._reset_num = 0
            self.close()
            self._start_env()
        self._obs = self.env.reset()
        return self._obs

    def step(self, raw_actions, **kwargs):
        timesteps = self.env.step(raw_actions, **kwargs)
        self._obs = timesteps
        self._rew = [timestep.reward for timestep in timesteps]
        self._done = True if timesteps[0].last() else False
        self._info = {'outcome': [0] * len(self.agent_players)}
        if self._done:
            self._info = {'outcome': self.get_outcome()}
        return self._obs, self._rew, self._done, self._info

    def close(self):
        if self.env:
            self.env.close()

    def get_outcome(self):
        outcome = [0] * len(self.agent_players)
        for i, o in enumerate(self._obs):
            player_id = o.observation.player_common.player_id
            for result in self.env._obs[i].player_result:
                if result.player_id == player_id:
                    outcome[i] = sc2_env.possible_results.get(result.result, 0)
        return outcome
