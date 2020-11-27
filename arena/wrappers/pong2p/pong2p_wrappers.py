from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import gym
import numpy as np

from collections import deque
from gym import spaces

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        if isinstance(reward, tuple):
            return tuple([np.sign(r) for r in reward])
        else:
            return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        if isinstance(self.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple([
                spaces.Box(low=0, high=255,
                    shape=(self.height, self.width, 1), dtype=np.uint8)
                for space in self.env.observation_space.spaces
            ])
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, observation):
        if isinstance(observation, tuple):
            return tuple([self._process_frame(obs) for obs in observation])
        else:
            return self._process_frame(observation)

    def _process_frame(self, frame):
        assert (frame.ndim == 3 and
               (frame.shape[2] == 3 or frame.shape[2] == 1) and
               frame.shape[0] == 210 and frame.shape[1] == 160)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        if isinstance(self.observation_space, spaces.Tuple):
            self.observation_space = spaces.Tuple([
                spaces.Box(low=0.0, high=1.0,
                    shape=space.shape, dtype=np.float32)
                for space in self.env.observation_space.spaces
            ])
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0,
                shape=self.env.observation_space.shape, dtype=np.float32)
                # shape=self.env.observation_space.spaces[0].shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        if isinstance(observation, tuple):
            return tuple([
                np.array(obs).astype(np.float32) / 255.0
                for obs in observation
            ])
        else:
            return np.array(observation).astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        if isinstance(env.observation_space, spaces.Tuple):
            self.frames = tuple([
                deque([], maxlen=k)
                for _ in env.observation_space.spaces
            ])
            self.observation_space = spaces.Tuple([
                spaces.Box(low=0, high=255, shape=(obs.shape[0], obs.shape[1], obs.shape[2] * k), dtype=np.uint8)
                for obs in env.observation_space.spaces
            ])
        else:
            self.frames = deque([], maxlen=k)
            shp = env.observation_space.shape
            self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            for frame, ob in zip(self.frames, obs):
                for _ in range(self.k):
                    frame.append(ob)
        else:
            for _ in range(self.k):
                self.frames.append(obs)

        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(obs, tuple):
            for frame, ob in zip(self.frames, obs):
                frame.append(ob)
        else:
            self.frames.append(obs)

        return self._get_ob(), reward, done, info

    def _get_ob(self):
        if isinstance(self.observation_space, spaces.Tuple):
            for frame in self.frames:
                assert len(frame) == self.k
            return tuple([
                np.array(LazyFrames(list(frame)))
                for frame in self.frames
            ])
        else:
            assert len(self.frames) == self.k
            return np.array(LazyFrames(list(self.frames)))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def wrap_pong(env_id, episode_life=True, clip_rewards=True, frame_stack=False, scale=False, seed=None):
    """Configure environment for Pong Selfplay.
    """
    env = gym.make(env_id)
    env.set_seed(seed)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

