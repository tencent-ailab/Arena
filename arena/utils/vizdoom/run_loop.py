from time import time

from arena.utils.vizdoom.player import player_window_cv
from arena.utils.vizdoom.core_env import VecEnv


def run_loop_venv(venv, agents, max_steps=3000, is_window_cv_visible=False,
                  verbose=4):
  obs = venv.reset()
  print('new episode')

  t_start = time()
  i_ep_step = 0
  ep_return = None
  for i_step in range(0, max_steps):
    actions = [ag.step(o) for ag, o in zip(agents, obs)]
    obs, rwd, dones, infos = venv.step(actions)
    print('run_loop/run_loop_venv/infos')
    print(infos)
    if ep_return is None:
      ep_return = rwd
    else:
      ep_return = [a+b for a, b in zip(ep_return, rwd)]

    if verbose >= 4:
      print('step: ', i_step)
      print('ep step: ', i_ep_step)
      for i, (o, r, d) in enumerate(zip(obs, rwd, dones)):
        print('o shape = {}, o type = {}, r = {}, done = {}'.format(
          o.shape, o.dtype, r, d))
        if is_window_cv_visible:
          player_window_cv(i, o)

    if all(dones):
      print('ep return when all dones: ', ep_return)
      venv.reset()
      i_ep_step = 0
      ep_return = None
      print('new episode')
    else:
      i_ep_step += 1

  t_end = time()
  print('elapsed_time = ', t_end - t_start)
  print('fps = ', float(max_steps) / float(t_end - t_start))


def run_loop_env(env, agent, **kwargs):
  venv = VecEnv([env])
  assert(type(agent) is not list)
  run_loop_venv(venv, [agent], **kwargs)
