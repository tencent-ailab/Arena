#!/usr/bin/python

"""Test script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from pysc2.env import sc2_env
from arena.utils.run_loop import run_loop
import importlib

FLAGS = flags.FLAGS
flags.DEFINE_string("player1", "Bot",
                     "Agent for player 1 ('Bot' for internal AI)")
flags.DEFINE_string("player2", "Bot",
                     "Agent for player 2 ('Bot' for internal AI)")
flags.DEFINE_string("difficulty", "A",
                     "Bot difficulty (from '1' to 'A')")
flags.DEFINE_integer("max_steps_per_episode", 10000,
                     "Max number of steps allowed per episode")
flags.DEFINE_integer("episodes", 0,
                     "Number of episodes (0 for infinity)")
flags.DEFINE_integer("screen_resolution", "640",
                     "Resolution for screen feature layers.")
flags.DEFINE_float("sleep_time_per_step", 0,
                     "Sleep time (in seconds) per step")
flags.DEFINE_float("screen_ratio", "1.33",
                     "Screen ratio of width / height")
flags.DEFINE_string("agent_interface_format", "feature",
                     "Agent Interface Format: [feature|rgb]")
flags.DEFINE_integer("minimap_resolution", "64",
                     "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")
flags.DEFINE_bool("merge_units_info", False, "Merge units info in timesteps.")
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_bool("visualize", False, "Visualize pygame screen")
flags.mark_flag_as_required("map")

def get_agent(agt_path):
    module, name = agt_path.rsplit('.', 1)
    agt_cls = getattr(importlib.import_module(module), name)
    return agt_cls()

def get_difficulty(level):
    diff_dict = { \
      "1": sc2_env.Difficulty.very_easy,
      "2": sc2_env.Difficulty.easy,
      "3": sc2_env.Difficulty.medium,
      "4": sc2_env.Difficulty.medium_hard,
      "5": sc2_env.Difficulty.hard,
      "6": sc2_env.Difficulty.hard,
      "7": sc2_env.Difficulty.very_hard,
      "8": sc2_env.Difficulty.cheat_vision,
      "9": sc2_env.Difficulty.cheat_money,
      "A": sc2_env.Difficulty.cheat_insane,
    }
    return diff_dict[level]

def main(unused_argv):
    """Run an agent."""
    step_mul = FLAGS.step_mul
    players = 2

    screen_res = (int(FLAGS.screen_ratio * FLAGS.screen_resolution)//4*4, FLAGS.screen_resolution)
    agent_interface_format = None
    if FLAGS.agent_interface_format == 'feature':
      agent_interface_format = sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=screen_res,
                    minimap=FLAGS.minimap_resolution))
    elif FLAGS.agent_interface_format == 'rgb':
      agent_interface_format = sc2_env.AgentInterfaceFormat(
                rgb_dimensions=sc2_env.Dimensions(
                    screen=screen_res,
                    minimap=FLAGS.minimap_resolution))
    else:
      raise NotImplementedError
    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
    agents = []
    bot_difficulty = get_difficulty(FLAGS.difficulty)
    if FLAGS.player1 == 'Bot':
      players[0] = sc2_env.Bot(sc2_env.Race.zerg, bot_difficulty)
    else:
      agents.append(get_agent(FLAGS.player1))
    if FLAGS.player2 == 'Bot':
      players[1] = sc2_env.Bot(sc2_env.Race.zerg, bot_difficulty)
    else:
      agents.append(get_agent(FLAGS.player2))
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            visualize=FLAGS.visualize,
            players=players,
            step_mul=step_mul,
            game_steps_per_episode=FLAGS.max_steps_per_episode * step_mul,
            agent_interface_format=agent_interface_format,
            disable_fog=FLAGS.disable_fog) as env:
        run_loop(agents, env,
            max_frames=0,
            max_episodes=FLAGS.episodes,
            sleep_time_per_step=FLAGS.sleep_time_per_step,
            merge_units_info=FLAGS.merge_units_info)

if __name__ == "__main__":
    app.run(main)
