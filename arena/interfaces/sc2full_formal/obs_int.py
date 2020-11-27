""" SC2 full formal Observation Interfaces"""
from copy import deepcopy
from collections import OrderedDict
import logging

import gym.spaces as spaces
from arena.interfaces.interface import Interface
from arena.interfaces.common import ActAsObsV2
import numpy as np

from timitate.lib6.pb2feature_converter import PB2FeatureConverter as PB2FeatureConverterV6
from timitate.lib6.pb2mask_converter import PB2MaskConverter as PB2MaskConverterV6
from timitate.utils.rep_db import unique_key_to_rep_info


class FullObsIntV7(Interface):
    def __init__(self, inter, zstat_data_src, input_map_resolution=(128, 128),
                 output_map_resolution=(128, 128), mmr=3500, game_version='4.10.0',
                 max_unit_num=600, max_bo_count=50, max_bobt_count=50,
                 zstat_presort_order_name=None, dict_space=False, zmaker_version='v4',
                 inj_larv_rule=False, ban_zb_rule=False, ban_rr_rule=False,
                 ban_hydra_rule=False, rr_food_cap=40, zb_food_cap=10,
                 hydra_food_cap=10, mof_lair_rule=False, hydra_spire_rule=False,
                 overseer_rule=False, expl_map_rule=False, baneling_rule=False,
                 add_cargo_to_units=False, crop_to_playable_area=False, ab_dropout_list=None):
        super(FullObsIntV7, self).__init__(inter)
        self.pb2feat = PB2FeatureConverterV6(map_resolution=input_map_resolution,
                                             zstat_data_src=zstat_data_src,
                                             max_bo_count=max_bo_count,
                                             max_bobt_count=max_bobt_count,
                                             game_version=game_version,
                                             dict_space=dict_space,
                                             zstat_version=zmaker_version,
                                             max_unit_num=max_unit_num,
                                             add_cargo_to_units=add_cargo_to_units,
                                             crop_to_playable_area=crop_to_playable_area)
        self.feat_spec = self.pb2feat.space
        self.pb2mask = PB2MaskConverterV6(map_resolution=output_map_resolution,
                                          game_version=game_version,
                                          dict_space=dict_space,
                                          max_unit_num=max_unit_num,
                                          inj_larv_rule=inj_larv_rule,
                                          ban_zb_rule=ban_zb_rule,
                                          ban_rr_rule=ban_rr_rule,
                                          ban_hydra_rule=ban_hydra_rule,
                                          rr_food_cap=rr_food_cap,
                                          zb_food_cap=zb_food_cap,
                                          hydra_food_cap=hydra_food_cap,
                                          mof_lair_rule=mof_lair_rule,
                                          hydra_spire_rule=hydra_spire_rule,
                                          overseer_rule=overseer_rule,
                                          expl_map_rule=expl_map_rule,
                                          baneling_rule=baneling_rule,
                                          add_cargo_to_units=add_cargo_to_units,
                                          ab_dropout_list=ab_dropout_list)
        self.mask_spec = self.pb2mask.space
        self._arg_mask = self.pb2mask.get_arg_mask()
        self.pb = None
        self._last_tar_tag = None
        self._last_units = None
        self._last_selected_unit_tags = None
        self.mmr = mmr
        self._zstat_presort_order_name = zstat_presort_order_name
        self._dict_space = dict_space
        self._max_unit_num = max_unit_num

    def reset(self, obs, **kwargs):
        super(FullObsIntV7, self).reset(obs, **kwargs)
        self._last_tar_tag = None
        self._last_units = None
        ###################
        # VERY DANGEROUS, be careful
        # map_name = 'KairosJunction'  # for temp debugging
        map_name = obs.game_info.map_name
        ###################
        start_pos = obs.game_info.start_raw.start_locations[0]
        # get the (zstat) zeroing probability
        if 'zeroing_prob' not in kwargs:
            logging.info('FullObsIntV5.reset: no zeroing_prob, defaults to 0.0')
            zstat_zeroing_prob = 0.0
        else:
            zstat_zeroing_prob = kwargs['zeroing_prob']
        # get the distribution
        if 'distrib' not in kwargs:
            logging.info('FullObsIntV5.reset: no distrib, defaults to None')
            distrib = None
        else:
            distrib = kwargs['distrib']
        # get the zstat category
        if 'zstat_category' not in kwargs:
            logging.info(
                'FullObsIntV5.reset: no zstat_category, defaults to None')
            zstat_category = None
        else:
            zstat_category = kwargs['zstat_category']
        # get the concrete zstat
        replay_name, player_id = self._sample_replay(
            distrib=distrib,
            zstat_presort_order=self._zstat_presort_order_name,
            zstat_category=zstat_category
        )
        # book-keep it to the root interface
        self.unwrapped().cur_zstat_fn = '{}-{}'.format(replay_name, player_id)
        self.pb2feat.reset(replay_name=replay_name, player_id=player_id,
                           mmr=self.mmr, map_name=map_name,
                           start_pos=(start_pos.x, start_pos.y),
                           zstat_zeroing_prob=zstat_zeroing_prob)
        self.pb2mask.reset()
        self._last_selected_unit_tags = None

    def _sample_replay(self, distrib, zstat_presort_order, zstat_category):
        logging.info('FullObsIntV5._sample_reply: zstat_presort_order_name={}'.format(
            zstat_presort_order))
        # check consistency
        if zstat_presort_order is not None and zstat_category is not None:
            raise ValueError('zstat_presort_order and zstat_category cannot be used simultaneously.')
        # decide the replay names from which we really sample
        if zstat_presort_order:
            all_replay_names = self.pb2feat.tarzstat_maker.zstat_keys_index.get_keys_by_presort_order(
                presort_order_name=zstat_presort_order
            )
        elif zstat_category:
            all_replay_names = self.pb2feat.tarzstat_maker.zstat_keys_index.get_keys_by_category(
                category_name=zstat_category
            )
        else:
            all_replay_names = self.pb2feat.tarzstat_maker.zstat_db.keys()
        # decide the distribution and do the sampling accordingly
        if distrib is None:
            logging.info('FullObsIntV5._sample_reply: distrib is None, defaults to uniform distribution.')
            distrib = np.ones(shape=(len(all_replay_names),), dtype=np.float32)
        p = distrib / distrib.sum()
        assert len(p) == len(all_replay_names), 'n={}, no. replays={}'.format(
            len(p), len(all_replay_names)
        )
        key = np.random.choice(all_replay_names, p=p)
        logging.info('FullObsIntV5._sample_reply: sampled from n={} replay files'.format(p.size))
        logging.info('FullObsIntV5._sample_reply: distrib={}'.format(p))
        replay_name, player_id = unique_key_to_rep_info(key)
        return replay_name, player_id

    @property
    def observation_space(self):
        if self._dict_space:
            obs_spec = spaces.Dict(
                OrderedDict(list(self.feat_spec.spaces.items())
                            + list(self.mask_spec.spaces.items())))
        else:
            obs_spec = spaces.Tuple(
                self.feat_spec.spaces + self.mask_spec.spaces)
        return obs_spec

    def obs(self, feat, mask):
        if self._dict_space:
            for k in mask:
                feat[k] = mask[k]
            return feat
        else:
            return list(feat) + list(mask)

    def act_trans(self, act):
        # cache the act
        self._last_act = act
        # use the cached act to determine the updated last-target-tag
        pysc2_timestep = self.unwrapped()._obs
        if not self._dict_space:
            # new action space and new index
            ab_index = self._last_act[0]
            last_selected_indices = self._last_act[3]
            target_index = self._last_act[4]
        else:
            ab_index = self._last_act['A_AB']
            last_selected_indices = self._last_act['A_SELECT']
            target_index = self._last_act['A_CMD_UNIT']
        self._last_units = pysc2_timestep.observation.raw_data.units
        self._last_tar_tag = self._last_units[target_index].tag \
            if self._arg_mask[ab_index, 4-1] else None
        self._last_selected_unit_tags = (
            [] if not self._arg_mask[ab_index, 3-1] else
            [self._last_units[idx].tag
             for idx in last_selected_indices if idx != self._max_unit_num])
        # do the routine
        if self.inter:
            act = self.inter.act_trans(act)
        return act

    def obs_trans(self, raw_obs):
        if self.inter:
            obs = self.inter.obs_trans(raw_obs)
        else:
            obs = raw_obs

        pb = obs, obs.game_info # TODO: to be simplified
        self.pb = pb
        feat = self.pb2feat.convert(pb, self._last_tar_tag, self._last_units, self._last_selected_unit_tags)
        mask = self.pb2mask.convert(pb)
        return self.obs(feat, mask)


class ActAsObsSC2(ActAsObsV2):
    def __init__(self, inter, override=False):
        super(ActAsObsSC2, self).__init__(inter, override)
        self._game_loop = 0

    def reset(self, obs, **kwargs):
        super(ActAsObsSC2, self).reset(obs, **kwargs)
        self._game_loop = 0

    def obs_trans(self, obs):
        obs_old = obs
        if self.inter:
            obs_old = self.inter.obs_trans(obs)
        # obs = (obs_pb, game_info), remove game_info?
        game_loop = int(obs.observation.game_loop)
        self._action['A_NOOP_NUM'] = min(game_loop - self._game_loop - 1, 127)
        self._game_loop = game_loop
        return self.wrapper.observation_transform(obs_old, self._action)
