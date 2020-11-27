#!/usr/bin/env python
# adopted from run_loop.py in pysc2
"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib.features import Features
from arena.utils.unit_util import merge_units
import time
    

def run_loop(agents, env, max_frames=0, max_episodes=0, sleep_time_per_step=0, merge_units_info=False):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    total_episodes = 0
    start_time = time.time()
    result_stat = [0]*3 # n_draw, n_win, n_loss

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    #for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
    #    agent.setup(obs_spec, act_spec)

    try:
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                total_frames += 1
                if merge_units_info:
                    assert len(timesteps)==2
                    # Merge units from two timesteps to one
                    merge_units(timesteps[0].observation.raw_data.units,
                        timesteps[1].observation.raw_data.units)
                    for i in range(2):
                        timesteps[i].observation['units'] = \
                            Features.transform_unit_control(timesteps[i].observation.raw_data.units)
                actions = [agent.step(timestep)
                           for agent, timestep in zip(agents, timesteps)]
                if max_frames and total_frames >= max_frames:
                    return
                if timesteps[0].last(): # player 1
                    result_stat[timesteps[0].reward] += 1
                    break
                if sleep_time_per_step > 0:
                    time.sleep(sleep_time_per_step)
                timesteps = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        print("Game result statistics: Win: %2d, Loss: %2d, Draw: %2d" % (
            result_stat[1], result_stat[-1], result_stat[0]))
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))
