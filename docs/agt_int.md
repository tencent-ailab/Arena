# AgtInt (Agent Interface)
AgtInt is a class to define action space/wrapper and observation space/wrapper
between agent and environment. After defining the Agent Interfaces of all the agent,
one can use
```
env_new = EnvWrapper(env, (agt_int_1, ... , agt_int_n))
```
to generate a new env with the desired action/observation space (usually simpler) for each agent.
Here env is a gym-style "multi-player" environment and agt_int_i is the Agent
Interface for each agent.

One can also use
```
agent_new = AgentWrapper(agent, agt_int)
```
to transform a agent in simpler action/observation space to a new agent which can
interact with raw environment directly.


## Basic idea of AgtInt
A AgtInt need to clearly define 'obs_spec' and 'action_spec' for the agent and also two
transformation functions:

* "observation_transform" transforms the raw observation into the desired simple observation

* "action_transform" transforms the agent's simple action into raw action in origin env

## AgtIntWrapper
AgtIntWrapper is used to transform AgtInt in a modular way:
```
agt_int = AgtInt()
agt_int = Discre4M2AWrapper(agt_int)
agt_int = UnitAttrWrapper(agt_int, override=True)
```
A AgtIntWrapper can override or modify the observation_transform/action_transform in agt_int,
obs_spec and action_spec also need specified.

# "Multi-player" Environment (env/base_env.py)
In this repository, we define a "multi-player" environment in following convention:

* observation space: a gym.spaces.Tuple object whose each entry means the obs_spec
for each agent.

* action space: a gym.spaces.Tuple object with the same length of observation space.
Each entry means the corresponding agent's action_spec. (A agent's action_spec may also
be a gym.spaces.Tuple object with each entry means the action_spec of each unit controlled
by this agent.)

Even there is only one player in the environment, we still use the above "multi-player" environment
convention to define observation/action space but with length of 1.

## EnvWrapper (env/base_env.py)
EnvWrapper is used to transform "Multi-player" Environment with defined AgtInts for all the players:
```
def step(self, actions):
    raw_actions = self.action(actions)
    obs, rwd, done, info = self.env.step(raw_actions)
    new_obs = self.observation(obs)
    return new_obs, rwd, done, info
```

# BaseAgent (env/base_agent.py)
A BaseAgent is basically a agent with "step" function to predict action in its action_spec
given the observation in its obs_spec.

## AgentWrapper (env/base_agent.py)
AgentWrapper is used to transform a BaseAgent object with defined AgtInt:
```
def step(self, obs):
    super(AgentWrapper, self).step(obs)
    obs = self.agt_int.observation_transform(obs)
    action = self.agent.step(obs)
    action = self.agt_int.action_transform(action)
    return action
```

