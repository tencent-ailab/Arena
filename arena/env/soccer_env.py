""" Arena compatible soccer env """
from dm_control.locomotion import soccer as dm_soccer
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from dm_env import specs
import pyglet
import sys
import cv2
from arena.utils.spaces import NoneSpace
from arena.interfaces.combine import Combine

class DmControlViewer:
    def __init__(self, width, height, depth=False):
        self.window = pyglet.window.Window(width=width, height=height, display=None)
        self.width = width
        self.height = height
        self.depth = depth

        if depth:
            self.format = 'RGB'
            self.pitch = self.width * -3
        else:
            self.format = 'RGB'
            self.pitch = self.width * -3

    def update(self, pixel):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        if self.depth:
            pixel = np.dstack([pixel.astype(np.uint8)] * 3)
        pyglet.image.ImageData(self.width, self.height, self.format, pixel.tobytes(), pitch=self.pitch).blit(0, 0)
        self.window.flip()

    def close(self):
        self.window.close()

class soccer_gym(core.Env):
    def __init__(self, team_size = 2, time_limit=45, disable_walker_contacts=True, team_num=2, render_name="human"):
        self.team_size = team_size
        self.team_num = team_num
        self.env = dm_soccer.load(self.team_size, time_limit, disable_walker_contacts)
        ac_sp_i = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        ac_sp = spaces.Tuple([spaces.Tuple(tuple([ac_sp_i]*self.team_size))]*self.team_num)
        self.action_space = ac_sp
        #print(self.action_space)
        self.observation_space = spaces.Tuple([NoneSpace(), NoneSpace()])
        self.timestep = None
        odict_sp = {}
        odict = self.env.observation_spec()
        for key in odict[0]:
            odict_sp[key] = spaces.Box(-np.inf, np.inf, shape=(np.int(np.prod(odict[0][key].shape)),))
        self.observation_space = spaces.Tuple([spaces.Tuple([spaces.Dict(odict_sp)]*self.team_size)]*self.team_num)
        # render
        render_mode_list = self.create_render_mode(render_name, show=False, return_pixel=True)
        if render_mode_list is not None:
            self.metadata['render.modes'] = list(render_mode_list.keys())
            self.viewer = {key:None for key in render_mode_list.keys()}
        else:
            self.metadata['render.modes'] = []
        self.render_mode_list = render_mode_list
        # set seed
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _team_obs_trans(self, timestep_obs):
        obs = []
        for i in range(self.team_num):
            obs_t = []
            for j in range(self.team_size):
                indx = i * self.team_size + j
                obs_t.append(timestep_obs[indx])
            obs.append(obs_t)
        return obs

    def reset(self):
        self.timestep = self.env.reset()
        obs = self._team_obs_trans(self.timestep.observation)
        return obs

    def step(self, a):
        # team actions
        act = []
        for i in range(self.team_num):
            act.extend(a[i])
        act = np.clip(act, -1., 1.)
        self.timestep = self.env.step(act)
        r, obs, info = [], [], []
        for i in range(self.team_num):
            r_t, obs_t, info_t = [], [], []
            for j in range(self.team_size):
                ar = []
                indx = i * self.team_size + j
                ar.append(float(self.timestep.observation[indx]["stats_home_score"]))
                ar.append(-1. * float(self.timestep.observation[indx]["stats_away_score"]))
                ar.append(float(self.timestep.observation[indx]["stats_vel_to_ball"]))
                ar.append(float(self.timestep.observation[indx]["stats_vel_ball_to_goal"]))
                ainfo = [int(self.timestep.observation[indx]["stats_home_score"]), int(self.timestep.observation[indx]["stats_away_score"])]
                r_t.append(ar)
                obs_t.append(self.timestep.observation[indx])
                info_t.append(ainfo)
            obs.append(obs_t)
            r.append(np.mean(np.array(r_t),0).tolist())
            info.append(info_t)
        return obs, r, self.timestep.last(), info

    def create_render_mode(self, name, show=True, return_pixel=False, height=480, width=640, camera_id=0, overlays=(),
             depth=False, scene_option=None):
        render_mode_list = {}
        render_kwargs = { 'height': height, 'width': width, 'camera_id': camera_id,
                                'overlays': overlays, 'depth': depth, 'scene_option': scene_option}
        render_mode_list[name] = {'show': show, 'return_pixel': return_pixel, 'render_kwargs': render_kwargs}
        return render_mode_list

    def render(self, mode='human', close=False):
        self.pixels = self.env.physics.render(**self.render_mode_list[mode]['render_kwargs'])
        if close:
            if self.viewer[mode] is not None:
                self._get_viewer(mode).close()
                self.viewer[mode] = None
            return
        elif self.render_mode_list[mode]['show']:
            self._get_viewer(mode).update(self.pixels)

        if self.render_mode_list[mode]['return_pixel']:
            #return self.pixels
            frame = self.pixels
            cv2.imshow('demo', frame)
            cv2.waitKey(100)
            return

    def _get_viewer(self, mode):
        if self.viewer[mode] is None:
            self.viewer[mode] = DmControlViewer(self.pixels.shape[1], self.pixels.shape[0], self.render_mode_list[mode]['render_kwargs']['depth'])
        return self.viewer[mode]

def main():
    team_size = 2
    env = soccer_gym(team_size, time_limit=45.)

    from arena.env.env_int_wrapper import EnvIntWrapper
    from arena.interfaces.soccer.obs_int import ConcatObsAct, Dict2Vec
    
    inter1 = Combine(None, [Dict2Vec(None), Dict2Vec(None)])
    inter1 = ConcatObsAct(inter1)
    inter2 = Combine(None, [Dict2Vec(None), Dict2Vec(None)])
    inter2 = ConcatObsAct(inter2)
    env = EnvIntWrapper(env, [inter1, inter2])
    state = env.reset()
    done = False
    print(env.observation_space)
    print(env.action_space)
    for t in range(10):
        all_acts = env.action_space.sample()
        observation, reward, done, info = env.step(all_acts)

if __name__ == '__main__':
    main()