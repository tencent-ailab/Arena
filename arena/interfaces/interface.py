#!/usr/bin/env python
# -*- coding:utf-8 -*-
from arena.interfaces.raw_int import RawInt


class Interface(RawInt):
    """
        Interface class

    """
    inter = None

    def __init__(self, interface):
        """ Initialization.
            
            :param  interface      previous interface to wrap on"""
        # In `__init__()` of derived class, one should firstly call
        #   super(self.__class__, self).__init__(interface)
        if interface is None:
            self.inter = RawInt()
        else:
            self.inter = interface
        assert isinstance(self.inter, RawInt)

    def reset(self, obs, **kwargs):
        """ Reset this interface.
            For some reasons, obs space and action space may be specified on reset().
            
            :param  obs            input obs (received by the root interface)"""
        # In `reset()` of derived class, one should firstly call
        #   super(self.__class__, self).reset(obs)
        self.inter.reset(obs, **kwargs)

    def obs_trans(self, obs):
        """ Observation Transformation. This is a recursive call. """
        obs = self.inter.obs_trans(obs)
        # Implement customized obs_trans here in derived class
        return obs

    def act_trans(self, act):
        """ Action Transformation. This is a recursive call. """
        # TODO(peng): raise NotImplementedError, encourage recursive call in derived class
        # Implement customized act_trans here in derived class
        act = self.inter.act_trans(act)
        return act

    def unwrapped(self):
        """ Get the root instance.
            This is usually used for storing global information.
            For example, raw obs and raw act are saved by RawInt(). """
        return self.inter.unwrapped()

    @property
    def observation_space(self):
        """ Observation Space, calculated in a recursive manner.
            Implement customized observation_space here in derived class. """
        return self.inter.observation_space

    @property
    def action_space(self):
        """ Action Space, calculated in a recursive manner.
            Implement customized action_space here in derived class """
        return self.inter.action_space

    def setup(self, observation_space, action_space):
        self.unwrapped().setup(observation_space, action_space)

    def __str__(self):
        """ Get the name of all stacked interface. """
        # TODO(peng): return my_name + '<' + wrapped_interface_name + '>'
        s = str(self.inter)
        return s+'<'+self.__class__.__name__+'>'
