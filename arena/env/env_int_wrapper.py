import numpy as np
from gym import Wrapper
from gym import spaces

class EnvIntWrapper(Wrapper):
    """ Environment Interface Wrapper """
    inters = []    # list of Interface

    def __init__(self, env, inters=()):
        super(EnvIntWrapper, self).__init__(env) # do we need this?
        self.env = env
        assert len(inters) == len(self.env.action_space.spaces)
        self.inters = inters
        self.__update_obs_and_act_space()

    def __update_obs_and_act_space(self):
        obs_space = []
        act_space = []
        for i, inter in enumerate(self.inters):
            if inter is not None:
                obs_space.append(inter.observation_space)
                act_space.append(inter.action_space)
            else:
                obs_space.append(self.env.observation_space.spaces[i])
                act_space.append(self.env.action_space.spaces[i])
        # These two spaces are required by gym
        self.observation_space = spaces.Tuple(obs_space)
        self.action_space = spaces.Tuple(act_space)

    def __act_trans(self, acts):
        assert len(acts) == len(self.env.action_space.spaces)
        rets = []
        for i, inter in enumerate(self.inters):
            if inter is not None:
                rets.append(self.inters[i].act_trans(acts[i]))
            else:
                rets.append(acts[i])
        return rets

    def __obs_trans(self, obss):
        assert len(obss) == len(self.env.observation_space.spaces)
        rets = []
        for i, inter in enumerate(self.inters):
            if inter is not None:
                rets.append(self.inters[i].obs_trans(obss[i]))
            else:
                rets.append(obss[i])
        return rets

    def reset(self, **kwargs):
        inter_kwargs = [{}] * len(self.inters)
        #if hasattr(kwargs, 'inter_kwargs'):
        if 'inter_kwargs' in kwargs:
            assert len(kwargs['inter_kwargs']) == len(self.inters), '{}, {}'.format(len(kwargs['inter_kwargs']),
                                                                                    len(self.inters))
            inter_kwargs = kwargs.pop('inter_kwargs')
        obs = self.env.reset(**kwargs)
        for i, inter in enumerate(self.inters):
            if inter is not None:
                inter.setup(self.env.observation_space.spaces[i],
                            self.env.action_space.spaces[i])
                inter.reset(obs[i], **inter_kwargs[i])
        self.__update_obs_and_act_space()
        return self.__obs_trans(obs)


    def step(self, acts, **kwargs):
        assert len(acts) == len(self.env.action_space.spaces)
        a = self.__act_trans(acts)
        obs, rwd, done, info = self.env.step(a, **kwargs)
        s = self.__obs_trans(obs)
        return s, rwd, done, info

    def close(self):
        self.env.close()


class SC2EnvIntWrapper(EnvIntWrapper):
    def __init__(self, env, inters=(), noop_fns=lambda x: 1,):
        super(SC2EnvIntWrapper, self).__init__(env, inters)
        if callable(noop_fns):
            self.noop_fns = [noop_fns] * len(inters)
        else:
            assert isinstance(noop_fns, list) or isinstance(noop_fns, tuple)
            assert len(noop_fns) == len(inters)
            assert all(callable(fn) for fn in noop_fns)
        self._noop_steps = np.array([0] * len(inters), dtype=np.int32)

    def reset(self, **kwargs):
        self._noop_steps = np.zeros_like(self._noop_steps)
        return super(SC2EnvIntWrapper, self).reset(**kwargs)

    def step(self, acts, **kwargs):
        predict_noop_steps = [0 if act is None else noop_fn(act)
                              for act, noop_fn in zip(acts, self.noop_fns)]
        self._noop_steps += np.array(predict_noop_steps, dtype=np.int32)
        min_noop_step = np.min(self._noop_steps)
        self.unwrapped.env._step_mul = min_noop_step
        self._noop_steps -= min_noop_step
        return super(SC2EnvIntWrapper, self).step(acts, **kwargs)
