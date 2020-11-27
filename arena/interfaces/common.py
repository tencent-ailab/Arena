from collections import deque, OrderedDict

from gym import spaces
import numpy as np

from arena.interfaces.interface import Interface
from arena.utils.spaces import NoneSpace


class AppendObsInt(Interface):
    '''Basic obs interface, append the new obs'''

    def __init__(self, inter, override=True):
        super(AppendObsInt, self).__init__(inter)
        self.wrapper = None
        self.override = override

    def reset(self, obs, **kwargs):
        super(AppendObsInt, self).reset(obs, **kwargs)
        if not self.override:
            assert (isinstance(self.inter.observation_space, spaces.Tuple)
                    or isinstance(self.inter.observation_space, spaces.Dict))
        try:
            self.build_wrapper()
        except NotImplementedError:
            pass
        except Exception as e:
            raise e

    def build_wrapper(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        if self.wrapper:
            return self.wrapper.observation_space
        else:
            return NoneSpace()

    def obs_trans(self, obs):
        self._obs = obs
        obs_pre = obs
        if self.inter:
            obs_pre = self.inter.obs_trans(obs)
        if not isinstance(obs_pre, tuple):
            obs_pre = tuple(obs_pre)
        return self.wrapper.observation_transform(obs_pre, self.unwrapped()._obs)


class ActionSeqFeature(object):
    def __init__(self, action_space, seq_len):
        if isinstance(action_space, spaces.Tuple):
            assert all([isinstance(space, spaces.Discrete)
                        or isinstance(space, spaces.MultiDiscrete)
                        for space in action_space.spaces])
            self._dims = np.concatenate([[space.n] if isinstance(space, spaces.Discrete)
                                         else space.nvec for space in action_space.spaces])
        elif isinstance(action_space, spaces.Discrete):
            self._dims = [action_space.n]
        else:
            assert isinstance(action_space, spaces.MultiDiscrete)
            self._dims = action_space.nvec
        self._action_space = action_space
        self._sum_dims = sum(self._dims)
        self._action_seq = [np.zeros(self._sum_dims, dtype=np.float32)] * seq_len

    def multi_one_hot(self, action):
        one_hot_act = np.zeros(self._sum_dims, dtype=np.float32)
        start = 0
        for dim, act in zip(self._dims, action):
            assert act < dim
            if act >= 0:
                one_hot_act[start + act] = 1
            start += dim
        return one_hot_act

    def reset(self):
        self._action_seq = [np.zeros(self._sum_dims, dtype=np.float32)] * len(self._action_seq)

    def push_action(self, action):
        self._action_seq.pop(0)
        if (isinstance(action, list) or isinstance(action, tuple)
            or (isinstance(action, np.ndarray) and len(action.shape) == 1)):
            self._action_seq.append(self.multi_one_hot(action))
        else:
            self._action_seq.append(self.multi_one_hot([action]))

    def features(self):
        return np.concatenate(self._action_seq)

    @property
    def num_dims(self):
        return self._sum_dims * len(self._action_seq)


class RemoveTupleObs(Interface):
    def reset(self, obs, **kwargs):
        super(RemoveTupleObs, self).reset(obs, **kwargs)
        assert isinstance(self.inter.observation_space, spaces.Tuple)
        assert len(self.inter.observation_space.spaces) == 1

    def obs_trans(self, obs):
        return self.inter.obs_trans(obs)[0]

    @property
    def observation_space(self):
        if isinstance(self.inter.observation_space, NoneSpace):
            return NoneSpace()
        else:
            assert isinstance(self.inter.observation_space, spaces.Tuple)
            return self.inter.observation_space.spaces[0]


class ConcatVecWrapper(Interface):
    def obs_trans(self, obs):
        return np.concatenate(self.inter.obs_trans(obs))

    @property
    def observation_space(self):
        if isinstance(self.inter.observation_space, NoneSpace):
            return NoneSpace()
        assert isinstance(self.inter.observation_space, spaces.Tuple)
        low = np.concatenate([box.low for box in self.inter.observation_space.spaces])
        high = np.concatenate([box.high for box in self.inter.observation_space.spaces])
        return spaces.Box(low=low, high=high)


class MultiBinFeature(object):
    def __init__(self, max_steps, n_bins):
        self.max_steps = max_steps
        self.n_bins = n_bins

    def features(self, steps):
        features = []
        for n_bin in self.n_bins:
            features.append(self._onehot(steps, n_bin))
        return np.concatenate(features)

    def _onehot(self, value, n_bin):
        bin_width = self.max_steps // n_bin
        feature = np.zeros(n_bin, dtype=np.float32)
        idx = int(value // bin_width)
        idx = n_bin - 1 if idx >= n_bin else idx
        feature[idx] = 1.0
        return feature

    @property
    def num_dims(self):
        return sum(self.n_bins)


class MultiBinObsInt(AppendObsInt):
    class MultiBinObsFunc(object):
        def __init__(self, max_steps, n_bins, func, override, space_old):
            self._progress_feature = MultiBinFeature(max_steps, n_bins)
            self._func = func
            n_dims = self._progress_feature.num_dims
            obs_space = spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((obs_space,))
            else:
                self.observation_space = spaces.Tuple(space_old.spaces + (obs_space,))

        def observation_transform(self, obs_old, obs):
            progress_feat = self._progress_feature.features(self._func(obs))
            return (progress_feat,) if self.override else obs_old + (progress_feat,)

    def __init__(self, inter, func, override=True, max_steps=800, n_bins=(80, 16, 4)):
        super(MultiBinObsInt, self).__init__(inter, override)
        self.max_steps = max_steps
        self.n_bins = n_bins
        self.func = func

    def reset(self, obs, **kwargs):
        super(MultiBinObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.MultiBinObsFunc(self.max_steps, self.n_bins,
                                            self.func, self.override,
                                            space_old=self.inter.observation_space)


class ActAsObs(AppendObsInt):
    class ActAsObsFunc(object):
        def __init__(self, action_space, n_action, override, space_old):
            self._action_seq_feature = ActionSeqFeature(action_space, n_action)
            n_dims = self._action_seq_feature.num_dims
            observation_space = spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((observation_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (observation_space,))

        def observation_transform(self, obs_old, action):
            if action is not None:
                self._action_seq_feature.push_action(action)
            action_seq_feat = self._action_seq_feature.features()
            return (action_seq_feat,) if self.override else obs_old + (action_seq_feat,)

    def __init__(self, inter, override=True, n_action=1):
        super(ActAsObs, self).__init__(inter, override)
        self.n_action = n_action
        self._action = None

    def reset(self, obs, **kwargs):
        super(ActAsObs, self).reset(obs, **kwargs)
        self.wrapper = self.ActAsObsFunc(self.inter.action_space,
                                         n_action=self.n_action,
                                         override=self.override,
                                         space_old=self.inter.observation_space)

    def act_trans(self, action):
        self._action = action
        if self.inter:
            action = self.inter.act_trans(action)
        return action

    def obs_trans(self, obs):
        self._obs = obs
        obs_old = obs
        if self.inter:
            obs_old = self.inter.obs_trans(obs)
        return self.wrapper.observation_transform(obs_old, self._action)


class ActAsObsV2(AppendObsInt):
    class ActAsObsFunc(object):
        def __init__(self, action_space, override, space_old):
            self.override = override
            self.iter_action = True
            if self.override:
                if isinstance(action_space, spaces.Tuple) or isinstance(action_space, spaces.Dict):
                    self.observation_space = action_space
                else:
                    self.iter_action = False
                    self.observation_space = spaces.Tuple([action_space])
            elif isinstance(space_old, spaces.Dict):
                assert isinstance(action_space, spaces.Dict)
                self.observation_space = \
                    spaces.Dict(OrderedDict(list(space_old.spaces.items()) + list(action_space.spaces.items())))
            elif isinstance(space_old, spaces.Tuple):
                assert not isinstance(action_space, spaces.Dict)
                if isinstance(action_space, spaces.Tuple):
                    self.observation_space = \
                        spaces.Tuple(tuple(space_old.spaces) + tuple(action_space.spaces))
                else:
                    self.iter_action = False
                    self.observation_space = \
                        spaces.Tuple(tuple(space_old.spaces) + (action_space,))
            else:
                raise TypeError('Observation space {} can not append.'.format(type(space_old)))

        def observation_transform(self, obs_old, action):
            if not self.iter_action:
                action = [action]
            if self.override:
                return action
            elif isinstance(self.observation_space, spaces.Tuple):
                return obs_old + (action,)
            else:
                for k, v in action.items():
                    obs_old[k] = v
                return obs_old

    def __init__(self, inter, override=False):
        super(ActAsObsV2, self).__init__(inter, override)
        self._action = None
        if (not isinstance(self.inter.observation_space, NoneSpace)
            and not isinstance(self.inter.action_space, NoneSpace)):
          self.wrapper = self.ActAsObsFunc(self.inter.action_space,
                                           override=self.override,
                                           space_old=self.inter.observation_space)

    def reset(self, obs, **kwargs):
        super(ActAsObsV2, self).reset(obs, **kwargs)
        self.wrapper = self.ActAsObsFunc(self.inter.action_space,
                                         override=self.override,
                                         space_old=self.inter.observation_space)
        action = self.inter.action_space.sample()
        if isinstance(self.inter.action_space, spaces.Tuple):
            self._action = [np.zeros_like(a) for a in action]
        elif isinstance(self.inter.action_space, spaces.Dict):
            self._action = OrderedDict([(name, np.zeros_like(a)) for name, a in action.items()])
        else:
            self._action = np.zeros_like(action)

    def act_trans(self, action):
        self._action = action
        if self.inter:
            action = self.inter.act_trans(action)
        return action

    def obs_trans(self, obs):
        obs_old = obs
        if self.inter:
            obs_old = self.inter.obs_trans(obs)
        return self.wrapper.observation_transform(obs_old, self._action)


def stack_space(space, k):
    assert isinstance(space, spaces.Box) or isinstance(space, spaces.Tuple)
    if isinstance(space, spaces.Box):
        return spaces.Box(low=np.array([space.low]*k),
                          high=np.array([space.high]*k),
                          dtype=space.dtype)
    else:
        return spaces.Tuple(tuple([stack_space(sp, k) for sp in space.spaces]))


def stack_frames(frames):
    if isinstance(frames[0], list) or isinstance(frames[0], tuple):
        return [stack_frames(f) for f in zip(*frames)]
    else:
        return np.array(frames)


class FrameStackInt(Interface):
    '''Frame stack interfce'''

    def __init__(self, inter, k):
        super(FrameStackInt, self).__init__(inter)
        self.frames = deque([], maxlen=k)
        self.k = k
        self.wrapper = None

    def reset(self, obs, **kwargs):
        super(FrameStackInt, self).reset(obs, **kwargs)
        self.frames.clear()

    @property
    def observation_space(self):
        inter_space = self.inter.observation_space
        if isinstance(inter_space, NoneSpace):
            return NoneSpace()
        else:
            return stack_space(inter_space, self.k)

    def obs_trans(self, obs):
        self._obs = obs
        obs_pre = self.inter.obs_trans(obs)
        self.frames.append(obs_pre)
        while len(self.frames) < self.k:
            self.frames.append(obs_pre)
        return stack_frames(self.frames)


class BoxTransformInt(Interface):
    '''Box Transform (including Reshape, Transpose, etc) interface'''
    ################
    ## index < 0 means self.inter.observation_space is Box, directly Transform
    ## index >= 0 means self.inter.observation_space is Tuple, Transform index_th space
    ################

    def __init__(self, inter, op, index=0):
        super(BoxTransformInt, self).__init__(inter)
        self.op = op
        self.index = index

    def reset(self, obs, **kwargs):
        super(BoxTransformInt, self).reset(obs, **kwargs)
        ob_space = self.inter.observation_space
        if self.index >= 0:
            assert isinstance(ob_space, spaces.Tuple)
            ob_space = ob_space.spaces[self.index]
        assert isinstance(ob_space, spaces.Box)

    @property
    def observation_space(self):
        inter_space = self.inter.observation_space
        if isinstance(inter_space, NoneSpace):
            return NoneSpace()
        elif isinstance(inter_space, spaces.Box):
            return spaces.Box(low=self.op(inter_space.low),
                              high=self.op(inter_space.high),
                              dtype=inter_space.dtype)
        else:
            assert isinstance(inter_space, spaces.Tuple)
            sps = list(inter_space.spaces)
            assert isinstance(sps[self.index], spaces.Box)
            sps[self.index] = spaces.Box(low=self.op(sps[self.index].low),
                                         high=self.op(sps[self.index].high),
                                         dtype=sps[self.index].dtype)
            return spaces.Tuple(tuple(sps))

    def obs_trans(self, obs):
        self._obs = obs
        obs_pre = self.inter.obs_trans(obs)
        if self.index >= 0:
            ob = list(obs_pre)
            ob[self.index] = self.op(ob[self.index])
            return tuple(ob)
        else:
            return self.op(obs_pre)


class ReshapeInt(BoxTransformInt):
    '''Reshape interfce, Reshape Box or Tuple(Box) with new_shape'''
    ################
    ## index < 0 means self.inter.observation_space is Box, directly Reshape
    ## index >= 0 means self.inter.observation_space is Tuple, Reshape index_th space
    ################

    def __init__(self, inter, new_shape, index=0):
        super(ReshapeInt, self).__init__(inter, lambda x: np.reshape(x, new_shape), index)


class TransoposeInt(BoxTransformInt):
    '''Transpose interfce, transpose Box or Tuple(Box) with axes'''
    ################
    ## index < 0 means self.inter.observation_space is Box, directly transpose
    ## index >= 0 means self.inter.observation_space is Tuple, transpose index_th space
    ################

    def __init__(self, inter, axes, index=0):
        super(TransoposeInt, self).__init__(inter, lambda x: np.transpose(x, axes), index)
