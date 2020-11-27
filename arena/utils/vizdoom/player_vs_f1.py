import numpy as np
import cv2

class PlayerHostConfig(object):
  def __init__(self, port, num_players=2):
    self.num_players = num_players
    self.port = port
    print('Host {}'.format(self.port))

class PlayerJoinConfig(object):
  def __init__(self):
    self.join_ip = 'localhost'
    # self.port = port
    # print('Player {}'.format(self.port))

class PlayerConfig(object):
  def __init__(self):
    self.config_path = None
    self.player_mode = None
    self.is_render_hud = None
    self.screen_resolution = None
    self.screen_format = None
    self.is_window_visible = None
    self.ticrate = None
    self.episode_timeout = None
    self.name = None
    self.colorset = None

    self.repeat_frame = 2
    self.num_bots = 0

    self.is_multiplayer_game = True
    self.host_cfg = None
    self.join_cfg = None

def player_host_setup(game, host_config):
  game.add_game_args(' '.join([
    "-host {}".format(host_config.num_players),
    "-port {}".format(host_config.port),
    "-netmode 0",
    "-deathmatch",
    "+sv_spawnfarthest 1",
    "+viz_nocheat 0",
  ]))
  return game

def player_join_setup(game, join_config):
  game.add_game_args(' '.join([
    "-join {}".format(join_config.join_ip),
    # "-port {}".format(join_config.port),
  ]))
  return game

def player_setup(game, player_config):
  pc = player_config  # a short name

  # read in the config from file first, allow over-write later
  if pc.config_path is not None:
    game.load_config(pc.config_path)

  if pc.player_mode is not None:
    game.set_mode(pc.player_mode)
  if pc.screen_resolution is not None:
    game.set_screen_resolution(pc.screen_resolution)
  if pc.screen_format is not None:
    game.set_screen_format(pc.screen_format)
  if pc.is_window_visible is not None:
    game.set_window_visible(pc.is_window_visible)
  if pc.ticrate is not None:
    game.set_ticrate(pc.ticrate)
  if pc.episode_timeout is not None:
    game.set_episode_timeout(pc.episode_timeout)

  game.set_console_enabled(False)

  if pc.name is not None:
    game.add_game_args("+name {}".format(pc.name))
  if pc.colorset is not None:
    game.add_game_args("+colorset {}".format(pc.colorset))
  return game

def player_window_cv(i_player, img, transpose=False):
  window_name = 'player ' + str(i_player)
  if transpose:
    img = np.transpose(img, axes=(1, 2, 0))
  cv2.imshow(window_name, img)
  cv2.waitKey(1)
