""" Arena compatible pommerman env """
import copy
import os
import pickle
import pommerman
import random
import time
import uuid
import numpy as np
from datetime import datetime
from pommerman import agents
from pommerman import helpers
from pommerman.constants import Item
from gym import Wrapper
from gym import spaces
from arena.utils.spaces import NoneSpace


class PommeBase(Wrapper):
    def __init__(self, env_id, random_side=True, agent_list=None,
                 rule_agents=[], replay_dir=None, n_player=4):
        self.n_player = n_player
        self.base_agents = [agents.RandomAgent() for _ in range(n_player)]
        if agent_list is None:
            self.agent_list = self.base_agents
        else:
            assert isinstance(agent_list, str)
            agent_list = agent_list.split(',')
            assert len(agent_list) == n_player
            self.agent_list = [
                helpers.make_agent_from_string(agent, i)
                for i, agent in enumerate(agent_list)
            ]
        # Make the environment using the agent list
        env = pommerman.make(env_id, self.agent_list)
        if agent_list is not None:
            for id_, agent in enumerate(self.base_agents):
                agent.init_agent(id_, env.spec._kwargs['game_type'])
        super(PommeBase, self).__init__(env)
        self.rule_agents = rule_agents
        self._random_side = random_side
        self.random_side()
        self._uuid = str(uuid.uuid1())[:8]
        self._replay_dir = replay_dir
        self._replay_data = {"mode": str(env_id)}


    def random_side(self):
        self.reverse_side = self._random_side and random.random() < 0.5
        agent_list = self.base_agents
        for rule_agent_id in self.rule_agents:
            id = (rule_agent_id + self.reverse_side) % self.n_player
            agent_list[id] = self.agent_list[id]
        self.env.set_agents(agent_list)

    def save_replay(self):
        if self._replay_dir is not None:
            path = os.path.join(self._replay_dir,
                                self._uuid + "-reverse_side_" +
                                str(self.reverse_side) +
                                "-" + str(datetime.now()) + ".pkl")
            print(f'Save replay to {path}, Reverse_side: {self.reverse_side}')
            with open(path, "wb") as file:
                pickle.dump(self._replay_data, file)
            return path

    def reset(self, **kwargs):
        self.random_side()
        state = super(PommeBase, self).reset()
        self._uuid = str(uuid.uuid1())[:8]
        self._replay_data["board"] = np.array(self.env._board,
                                              copy=True).tolist()
        self._replay_data["items"] = copy.deepcopy(self.env._items)
        self._replay_data["actions"] = []
        return self.obs_transform(state)

    def obs_transform(self, state):
        raise NotImplementedError

    def act_transform(self, actions):
        raise NotImplementedError

    def rwd_transform(self, r):
        raise NotImplementedError

    def step(self, actions):
        act = self.act_transform(actions)
        if self.rule_agents:
            inner_act = self.env.act(self.env.observations)
            for i in self.rule_agents:
                id = (i + self.reverse_side) % self.n_player
                act[id] = inner_act[id]
        state, r, done, info = self.env.step(act)
        self._replay_data["actions"].append(np.array(act, copy=True).tolist())
        if done:
            self._replay_data["reward"] = r
            self.save_replay()
            for i, agent in enumerate(self.env._agents):
                agent.episode_end(r[i])
        reward = self.rwd_transform(r)
        info['outcome'] = reward if reward[0] != reward[1] else [0, 0]
        info.pop('result')
        if 'winners' in info:
          info.pop('winners')
        return self.obs_transform(state), reward, done, info

    def close(self):
        for agent in self.agent_list:
            agent.shutdown()
        self.env.close()


class pommerman_1v1(PommeBase):
    def __init__(self, env_id='Pomme1v1', random_side=True,
                 agent_list=None, rule_agents=[], replay_dir=None):
        super(pommerman_1v1, self).__init__(env_id, random_side, agent_list,
                                            rule_agents, replay_dir, n_player=2)
        ac_sp = self.env.action_space
        self.action_space = spaces.Tuple([ac_sp] * 2)
        self.observation_space = spaces.Tuple([NoneSpace(), NoneSpace()])

    def obs_transform(self, state):
        if self.reverse_side:
            obs = [state[1], state[0]]
        else:
            obs = [state[0], state[1]]
        return obs

    def act_transform(self, actions):
        if self.reverse_side:
            act = [actions[1], actions[0]]
        else:
            act = [actions[0], actions[1]]
        return act

    def rwd_transform(self, r):
        if self.reverse_side:
            reward = [r[1], r[0]]
        else:
            reward = r
        return reward


class pommerman_2v2(PommeBase):
    def __init__(self, env_id='PommeTeam-v0', random_side=True,
                 agent_list=None, rule_agents=[], replay_dir=None):
        super(pommerman_2v2, self).__init__(env_id, random_side, agent_list,
                                            rule_agents, replay_dir, n_player=4)
        ac_sp = spaces.Tuple([spaces.Discrete(6)] * 2)
        self.action_space = spaces.Tuple([ac_sp] * 2)
        ob_sp = spaces.Tuple([NoneSpace(), NoneSpace()])
        self.observation_space = spaces.Tuple([ob_sp] * 2)

    def obs_transform(self, state):
        if self.reverse_side:
            obs = [(state[1], state[3]),
                   (state[0], state[2])]
        else:
            obs = [(state[0], state[2]),
                   (state[1], state[3])]
        return obs

    def act_transform(self, actions):
        if self.reverse_side:
            act = [actions[1][0], actions[0][0], actions[1][1], actions[0][1]]
        else:
            act = [actions[0][0], actions[1][0], actions[0][1], actions[1][1]]
        return act

    def rwd_transform(self, r):
        if self.reverse_side:
            reward = r[1:3]
        else:
            reward = r[0:2]
        return reward


class vec_rwd(Wrapper):
    def __init__(self, env):
        super(vec_rwd, self).__init__(env)

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        return self.obs

    @staticmethod
    def rwd_transform(reward, ob, ob_old):
        rwd_lives = [(ob[1]['teammate'].value in ob[1]['alive']) -
                       (ob_old[1]['teammate'].value in ob_old[1]['alive']),
                     (ob[0]['teammate'].value in ob[0]['alive']) -
                       (ob_old[0]['teammate'].value in ob_old[0]['alive']),
                     (ob_old[0]['enemies'][0].value in ob_old[0]['alive']) -
                       (ob[0]['enemies'][0].value in ob[0]['alive']),
                     (ob_old[0]['enemies'][1].value in ob_old[0]['alive']) -
                       (ob[0]['enemies'][1].value in ob[0]['alive'])]
        rwd_blast_strength = [ob[0]['blast_strength'] - ob_old[0]['blast_strength'],
                              ob[1]['blast_strength'] - ob_old[1]['blast_strength']]
        rwd_can_kick = [ob[0]['can_kick'] - ob_old[0]['can_kick'],
                        ob[1]['can_kick'] - ob_old[1]['can_kick']]
        rwd_ammo = [int(ob_old[0]['board'][ob[0]['position']] == Item.ExtraBomb.value),
                    int(ob_old[1]['board'][ob[1]['position']] == Item.ExtraBomb.value)]
        return tuple([reward] + rwd_lives + rwd_blast_strength + rwd_can_kick + rwd_ammo)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = [0, 0] if reward[0] == reward[1] else reward
        rwd = [self.rwd_transform(r, ob, ob_old)
               for r, ob, ob_old in zip(reward, obs, self.obs)]
        self.obs = obs
        return obs, rwd, done, info


def pommerman_replay(replay_path, n_player=4, render_fps=10):
    with open(replay_path, "rb") as file:
        replay_data = pickle.load(file)
    env = pommerman.make(replay_data['mode'],
                         [agents.BaseAgent() for _ in range(n_player)])
    env.reset()
    env._render_fps = render_fps
    env._board = np.array(replay_data["board"])
    env._items = replay_data["items"]
    reward = None
    for i in replay_data["actions"]:
        env.render()
        reward, done = env.step(i)[1:3]
        if done:
            env.render()
            time.sleep(10)
            break
    if reward != replay_data["reward"]:
        print(reward)
        raise Exception("The current reward doesn't match the expected reward")
    env.close()


def save2json(replay_path, n_player=4, agent_names=range(4)):
    with open(replay_path, "rb") as file:
        replay_data = pickle.load(file)
    from pommerman.utility import join_json_state
    path, name = os.path.split(replay_path)
    t = name.strip('.pkl')[-26:]
    print(name)
    finished_at = datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f").isoformat()
    record_json_dir = os.path.join(path, name.split('.')[0])
    if not os.path.exists(record_json_dir):
        os.mkdir(record_json_dir)
    env = pommerman.make(replay_data['mode'],
                         [agents.BaseAgent() for _ in range(n_player)])
    env.reset()
    env._board = np.array(replay_data["board"])
    env._items = replay_data["items"]
    reward = None
    for i in replay_data["actions"]:
        env.save_json(record_json_dir=record_json_dir)
        reward, done, info = env.step(i)[1:]
        if done:
            env.save_json(record_json_dir=record_json_dir)
            join_json_state(record_json_dir=record_json_dir,
                            agents=[str(i) for i in agent_names],
                            finished_at=finished_at,
                            config=replay_data['mode'], info=info)
            time.sleep(1)
            break
    if reward != replay_data["reward"]:
        print(reward)
        raise Exception("The current reward doesn't match the expected reward")
    env.close()


def main_1v1():
    env = pommerman_1v1(env_id='OneVsOne-v0', replay_dir='./')
    env.reset()
    t = 1
    import time
    done = False
    while not done:
        env.render()
        time.sleep(0.1)
        if t % 2 == 1:
            actions = (5,5)
        else:
            actions = (1,1)
        t += 1
        print(actions)
        state, reward, done, info = env.step(actions)
    env.close()


def main_2v2(version):
    env = pommerman_2v2(env_id='PommeTeamCompetition-v0', replay_dir='./')
    env = vec_rwd(env)
    from arena.env.env_int_wrapper import EnvIntWrapper
    if version == 'v1':
        from arena.interfaces.pommerman.obs_int import BoardMapObs, AttrObsInt, ActMaskObsInt, PosObsInt, CombineObsInt
        from arena.interfaces.common import ActAsObs, MultiBinObsInt, FrameStackInt, TransoposeInt, ReshapeInt, BoxTransformInt
        inter = CombineObsInt(None)
        inter = BoardMapObs(inter)
        inter = FrameStackInt(inter, 4)
        inter = TransoposeInt(inter, axes=(1,2,0,3), index=0)
        inter = ReshapeInt(inter, new_shape=(11, 11, -1), index=0)
        inter = ActAsObs(inter, override=False, n_action=10)
        inter = MultiBinObsInt(inter, lambda obs: obs['step_count'], override=False)
        inter = AttrObsInt(inter, override=False)
        inter = PosObsInt(inter, override=False)
        inter = ActMaskObsInt(inter)
    else:
        from arena.interfaces.pommerman.obs_int_v2 import BoardMapObs, \
            AttrObsInt, ActMaskObsInt, RotateInt
        from arena.interfaces.common import ActAsObs, MultiBinObsInt, \
            FrameStackInt, TransoposeInt, ReshapeInt
        from arena.interfaces.combine import Combine

        inter1 = RotateInt(None)
        inter1 = BoardMapObs(inter1, override=True)
        inter1 = AttrObsInt(inter1, override=False)
        inter1 = ActAsObs(inter1, override=False, n_action=2)
        inter1 = MultiBinObsInt(inter1, lambda obs: obs['step_count'],
                                override=False)
        inter1 = ActMaskObsInt(inter1, override=False)

        inter2 = RotateInt(None)
        inter2 = BoardMapObs(inter2, override=True)
        inter2 = AttrObsInt(inter2, override=False)
        inter2 = ActAsObs(inter2, override=False, n_action=2)
        inter2 = MultiBinObsInt(inter2, lambda obs: obs['step_count'],
                                override=False)
        inter2 = ActMaskObsInt(inter2, override=False)

        inter = Combine(None, [inter1, inter2])
    env = EnvIntWrapper(env, [inter, None])
    state = env.reset()
    done = False
    print(env.observation_space)
    print(env.action_space)
    t = 1
    import time
    while not done:
        env.render()
        time.sleep(0.1)
        if t % 2 == 1:
            actions = ([5,5],[5,5])
        else:
            actions = ([1,1],[1,1])
        t += 1
        print(actions)
        state, reward, done, info = env.step(actions)
    env.close()


def main():
    main_1v1()
    main_2v2('v1')
    main_2v2('v2')

if __name__ == '__main__':
    main()
