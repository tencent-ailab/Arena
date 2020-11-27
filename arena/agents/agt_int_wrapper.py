from arena.agents.base_agent import BaseAgent
from arena.interfaces.raw_int import RawInt


class AgtIntWrapper(BaseAgent):
    inter = None

    def __init__(self, agent, interface=RawInt(), step_mul=1):
        super(AgtIntWrapper, self).__init__()
        self.agent = agent
        self.inter = interface
        self.step_mul = step_mul
        self.act = None
        assert isinstance(self.inter, RawInt)

    def setup(self, observation_space, action_space):
        super(AgtIntWrapper, self).setup(observation_space, action_space)
        self.inter.unwrapped()._observation_space = observation_space
        self.inter.unwrapped()._action_space = action_space

    def reset(self, obs, inter_kwargs={}):
        super(AgtIntWrapper, self).reset(obs)
        self.inter.reset(obs, **inter_kwargs)
        self.agent.setup(self.inter.observation_space, self.inter.action_space)
        self.agent.reset(self.inter.obs_trans(obs))
        self.act = None

    def step(self, obs):
        super(AgtIntWrapper, self).step(obs)
        obs = self.inter.obs_trans(obs)
        if (self.steps -1) % self.step_mul == 0:
          self.act = self.agent.step(obs)
        act = self.inter.act_trans(self.act)
        return act
