
class BaseAgent(object):
    """ Base agent class """
    observation_space = None
    action_space = None

    def __init__(self):
        self.episodes = 0
        self.steps = 0
        self._obs = None

    def setup(self, observation_space, action_space):
        """ Set the observation space and action space

        Parameters:
            observation_space (gym.spaces.Space): Observation space
            action_space (gym.spaces.Space): Action space
        """
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, obs=None):
        """ Reset the agent with initial observation.

        Parameters:
            obs: Initial observation
        """
        self._obs = obs
        self.episodes += 1
        self.steps = 0

    def step(self, obs):
        """ Step the agent, observe the obs and return the action.

        Parameters:
            obs: Initial observation

        Returns:
            action of the agent
        """
        self._obs = obs
        self.steps += 1
        return None

