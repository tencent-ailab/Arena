import sys
import numpy as np
import gym
import pygame

from gym import error, spaces, utils
from gym.utils import seeding

from .pong2p_game import PongGame


class PongSinglePlayerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ball_speed=4, bat_speed=16, max_num_rounds=20, random_seed=None):
        SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3))
        self.action_space = spaces.Discrete(6)

        pygame.init()
        self._surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._viewer = None
        self._rng = np.random.RandomState()
        self._game = PongGame(
            has_double_players=False,
            window_size=(SCREEN_WIDTH, SCREEN_WIDTH),
            ball_speed=ball_speed,
            bat_speed=bat_speed,
            max_num_rounds=max_num_rounds)

    def set_seed(self, seed):
        self._game.set_seed(seed)

    def _seed(self, seed=None):
        self._rng.seed(seed)

    def step(self, action):
        #assert self.action_space.contains(action)
        bat_directions = [0, 1, 2, 3, 4, 5]
        rewards, done = self._game.step(bat_directions[action], None)
        obs = self._get_screen_img()
        return (obs, rewards[0], done, {})

    def reset(self):
        self._game.reset_game()
        obs = self._get_screen_img()
        return obs

    def _render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            pygame.quit()
            return
        img = self._get_screen_img()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img)

    def render(self, mode='human', close=False):
        return self._render(mode, close)

    def _get_screen_img(self):
        self._game.draw(self._surface)
        self._game.draw_scoreboard(self._surface)
        obs = self._surface_to_img(self._surface)
        return obs

    def _surface_to_img(self, surface):
        img = pygame.surfarray.array3d(surface).astype(np.uint8)
        return np.transpose(img, (1, 0, 2))

class PongDoublePlayerEnv(PongSinglePlayerEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, ball_speed=4, bat_speed=16, max_num_rounds=20, random_seed=None):
        SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
        self.observation_space = spaces.Tuple([
            spaces.Box(
                low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3)),
            spaces.Box(
                low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3))
        ])
        self.action_space = spaces.Tuple(
            [spaces.Discrete(6), spaces.Discrete(6)])

        pygame.init()
        self._surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self._viewer = None
        self._game = PongGame(
            has_double_players=True,
            window_size=(SCREEN_WIDTH, SCREEN_WIDTH),
            ball_speed=ball_speed,
            bat_speed=bat_speed,
            max_num_rounds=max_num_rounds)

    def step(self, action):
        #assert self.action_space.contains(action)
        left_player_action, right_player_action = action
        bat_directions = [0, 1, 2, 3, 4, 5]
        rewards, done = self._game.step(bat_directions[left_player_action],
                                        bat_directions[right_player_action])
        obs = self._get_screen_img_double_player()
        return (obs, rewards, done, {})

    def reset(self):
        self._game.reset_game()
        obs = self._get_screen_img_double_player()
        return obs

    def _get_screen_img_double_player(self):
        self._game.draw(self._surface)
        surface_flipped = pygame.transform.flip(self._surface, True, False)
        self._game.draw_scoreboard(self._surface)
        self._game.draw_scoreboard(surface_flipped)
        obs = self._surface_to_img(self._surface)
        obs_flip = self._surface_to_img(surface_flipped)
        return (obs, obs_flip)
