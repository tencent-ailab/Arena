#!/usr/bin/env python
# -*- coding:utf-8 -*-
from gym import spaces
from arena.interfaces.interface import Interface
from arena.interfaces.raw_int import RawInt


class Combine(Interface):
    """
        Concat several Interface to form a new Interface

    """
    inter = None

    def __init__(self, interface, sub_interfaces=[]):
        """ Initialization.
            
        :param  interface        previous interface to wrap on
        :param  sub_interfaces   interfaces to combine
        """

        if interface is None:
            self.inter = RawInt()
        else:
            self.inter = interface
        assert isinstance(self.inter, RawInt)

        self.sub_interfaces = list(sub_interfaces)
        for i, interface in enumerate(sub_interfaces):
            if interface is None:
                self.sub_interfaces[i] = RawInt()
            else:
                assert isinstance(interface, RawInt)

    def setup(self, observation_space, action_space):
        self.unwrapped().setup(observation_space, action_space)
        for i in range(len(self.sub_interfaces)):
            self.sub_interfaces[i].setup(observation_space.spaces[i],
                                         action_space.spaces[i])

    def reset(self, obs, **kwargs):
        inter_ob_sp = self.inter.observation_space
        inter_ac_sp = self.inter.action_space
        assert isinstance(inter_ob_sp, spaces.Tuple)
        assert isinstance(inter_ac_sp, spaces.Tuple)
        assert len(inter_ob_sp.spaces) == len(self.sub_interfaces)
        self.inter.reset(obs)
        for i in range(len(self.sub_interfaces)):
            self.sub_interfaces[i].setup(inter_ob_sp.spaces[i],
                                         inter_ac_sp.spaces[i])
            self.sub_interfaces[i].reset(obs[i])

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        sub_obs = tuple([sub_inter.obs_trans(ob)
                         for ob, sub_inter in zip(obs, self.sub_interfaces)])
        return self._obs_trans(obs, sub_obs)

    def _obs_trans(self, obs, sub_obs):
        """ Observation Transformation.
            obs is observation from self.inter
            sub_obs are observations from sub_interfaces"""
        return sub_obs

    def _act_trans(self, act):
        act = [sub_inter.act_trans(ac)
               for ac, sub_inter in zip(act, self.sub_interfaces)]
        return act

    @property
    def observation_space(self):
        return spaces.Tuple([inter.observation_space
                             for inter in self.sub_interfaces])

    @property
    def action_space(self):
        return spaces.Tuple([inter.action_space
                             for inter in self.sub_interfaces])

    def __str__(self):
        """ Get the name of all stacked interface. """
        s = str(self.inter)
        combine_s = str([str(inter) for inter in self.sub_interfaces])
        return s+'<'+self.__class__.__name__+combine_s+'>'
