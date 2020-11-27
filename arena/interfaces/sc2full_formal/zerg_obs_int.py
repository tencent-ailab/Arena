from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import UNIT_TYPEID, UPGRADE_ID
from arena.interfaces.common import AppendObsInt
from arena.utils.spaces import NoneSpace


class ZergTechObsInt(AppendObsInt):
    class Wrapper(object):
        def __init__(self, override, space_old):
            '''upgrade of self (enemy's upgrade is unavailable)'''
            self.tech_list = [UPGRADE_ID.BURROW.value,
                              UPGRADE_ID.CENTRIFICALHOOKS.value,
                              UPGRADE_ID.CHITINOUSPLATING.value,
                              UPGRADE_ID.EVOLVEMUSCULARAUGMENTS.value,
                              UPGRADE_ID.GLIALRECONSTITUTION.value,
                              UPGRADE_ID.INFESTORENERGYUPGRADE.value,
                              UPGRADE_ID.ZERGLINGATTACKSPEED.value,
                              UPGRADE_ID.ZERGLINGMOVEMENTSPEED.value,
                              UPGRADE_ID.ZERGFLYERARMORSLEVEL1.value,
                              UPGRADE_ID.ZERGFLYERARMORSLEVEL2.value,
                              UPGRADE_ID.ZERGFLYERARMORSLEVEL3.value,
                              UPGRADE_ID.ZERGFLYERWEAPONSLEVEL1.value,
                              UPGRADE_ID.ZERGFLYERWEAPONSLEVEL2.value,
                              UPGRADE_ID.ZERGFLYERWEAPONSLEVEL3.value,
                              UPGRADE_ID.ZERGGROUNDARMORSLEVEL1.value,
                              UPGRADE_ID.ZERGGROUNDARMORSLEVEL2.value,
                              UPGRADE_ID.ZERGGROUNDARMORSLEVEL3.value,
                              UPGRADE_ID.ZERGMELEEWEAPONSLEVEL1.value,
                              UPGRADE_ID.ZERGMELEEWEAPONSLEVEL2.value,
                              UPGRADE_ID.ZERGMELEEWEAPONSLEVEL3.value,
                              UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL1.value,
                              UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL2.value,
                              UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL3.value]
            observation_space = spaces.Box(0.0, 1.0, [len(self.tech_list)], dtype=np.float32)
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((observation_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (observation_space,))

        def observation_transform(self, obs_pre, obs):
            new_obs = [upgrade in obs.observation['raw_data'].player.upgrade_ids for upgrade in self.tech_list]
            new_obs = np.array(new_obs, dtype=np.float32)
            return [new_obs] if self.override else list(obs_pre) + [new_obs]

    def reset(self, obs, **kwargs):
        super(ZergTechObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.Wrapper(override=self.override,
                                    space_old=self.inter.observation_space)


class ZergUnitProg(object):
    def __init__(self, tech_tree, override, space_old,
                 building_list=None, tech_list=None, dtype=np.float32):
        '''Return (in_progress, progess) for each building and tech
        in_progress includes the period the ordered drone moving to target pos
        Only self, enemy's information not available'''
        self.TT = tech_tree
        self.dtype = dtype
        self.building_list = building_list or \
                             [UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                              UNIT_TYPE.ZERG_ROACHWARREN.value,
                              UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                              UNIT_TYPE.ZERG_HATCHERY.value,
                              UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                              UNIT_TYPE.ZERG_BANELINGNEST.value,
                              UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                              UNIT_TYPE.ZERG_SPIRE.value,
                              UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                              UNIT_TYPE.ZERG_LURKERDENMP.value,
                              UNIT_TYPE.ZERG_LAIR.value,
                              UNIT_TYPE.ZERG_HIVE.value,
                              UNIT_TYPE.ZERG_GREATERSPIRE.value]
        self.tech_list = tech_list or \
                         [UPGRADE_ID.BURROW.value,
                          UPGRADE_ID.CENTRIFICALHOOKS.value,
                          UPGRADE_ID.CHITINOUSPLATING.value,
                          UPGRADE_ID.EVOLVEMUSCULARAUGMENTS.value,
                          UPGRADE_ID.GLIALRECONSTITUTION.value,
                          UPGRADE_ID.INFESTORENERGYUPGRADE.value,
                          UPGRADE_ID.ZERGLINGATTACKSPEED.value,
                          UPGRADE_ID.ZERGLINGMOVEMENTSPEED.value,
                          UPGRADE_ID.ZERGFLYERARMORSLEVEL1.value,
                          UPGRADE_ID.ZERGFLYERARMORSLEVEL2.value,
                          UPGRADE_ID.ZERGFLYERARMORSLEVEL3.value,
                          UPGRADE_ID.ZERGFLYERWEAPONSLEVEL1.value,
                          UPGRADE_ID.ZERGFLYERWEAPONSLEVEL2.value,
                          UPGRADE_ID.ZERGFLYERWEAPONSLEVEL3.value,
                          UPGRADE_ID.ZERGGROUNDARMORSLEVEL1.value,
                          UPGRADE_ID.ZERGGROUNDARMORSLEVEL2.value,
                          UPGRADE_ID.ZERGGROUNDARMORSLEVEL3.value,
                          UPGRADE_ID.ZERGMELEEWEAPONSLEVEL1.value,
                          UPGRADE_ID.ZERGMELEEWEAPONSLEVEL2.value,
                          UPGRADE_ID.ZERGMELEEWEAPONSLEVEL3.value,
                          UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL1.value,
                          UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL2.value,
                          UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL3.value]
        n_dims = len(self.building_list) * 2 + len(self.tech_list) * 2
        observation_space = spaces.Box(0.0, 1.0, [n_dims], dtype=dtype)
        self.override = override
        if self.override or isinstance(space_old, NoneSpace):
            self.observation_space = spaces.Tuple((observation_space,))
        else:
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (observation_space,))
        self.morph_history = {}  # tag: [ability_id, game_loop_start, game_loop_now]

    def building_progress(self, unit_type, obs, alliance=1):
        in_progress = 0
        progress = 0
        unit_data = self.TT.getUnitData(unit_type)
        if not unit_data.isBuilding:
            print('building_in_progress can only be used for buildings!')
        game_loop = obs.observation.game_loop
        if isinstance(game_loop, np.ndarray):
            game_loop = game_loop[0]
        if unit_type in [UNIT_TYPE.ZERG_LAIR.value,
                         UNIT_TYPE.ZERG_HIVE.value,
                         UNIT_TYPE.ZERG_GREATERSPIRE.value]:
            builders = [unit for unit in obs.observation.raw_data.units
                        if unit.unit_type in unit_data.whatBuilds
                        and unit.alliance == alliance]
            for builder in builders:
                if len(builder.orders) > 0 and builder.orders[0].ability_id == unit_data.buildAbility:
                    # pb do not return the progress of unit morphing
                    if (builder.unit_type not in self.morph_history or
                            self.morph_history[builder.unit_type][0] != unit_data.buildAbility):
                        self.morph_history[builder.unit_type] = [unit_data.buildAbility, game_loop, game_loop]
                    else:
                        self.morph_history[builder.unit_type][2] = game_loop
                    in_progress = 1
                    progress = self.morph_history[builder.unit_type][2] - self.morph_history[builder.unit_type][1]
                    progress /= float(unit_data.buildTime)
        else:
            for unit in obs.observation.raw_data.units:
                if (unit.unit_type == unit_type
                        and unit.alliance == alliance
                        and unit.build_progress < 1):
                    in_progress = 1
                    progress = max(progress, unit.build_progress)
                if (unit.unit_type == UNIT_TYPEID.ZERG_DRONE.value
                    and unit.alliance == alliance
                    and len(unit.orders) > 0
                    and unit.orders[0].ability_id == unit_data.buildAbility):
                    in_progress = 1
        return in_progress, progress

    def update_morph_history(self, obs):
        # pb do not return the progress of unit morphing
        game_loop = obs.observation.game_loop
        if isinstance(game_loop, np.ndarray):
            game_loop = game_loop[0]
        for tag in self.morph_history:
            if self.morph_history[tag][2] != game_loop:
                self.morph_history[tag][0] = None

    def upgrade_progress(self, upgrade_type, obs, alliance=1):
        in_progress = 0
        progress = 0
        data = self.TT.getUpgradeData(upgrade_type)
        builders = [unit for unit in obs.observation.raw_data.units
                    if unit.unit_type in data.whatBuilds
                    and unit.alliance == alliance]
        for builder in builders:
            if len(builder.orders) > 0 and builder.orders[0].ability_id == data.buildAbility:
                in_progress = 1
                progress = builder.orders[0].progress
        return in_progress, progress

    def observation_transform(self, obs_pre, obs):
        new_obs = []
        for building in self.building_list:
            new_obs.extend(self.building_progress(building, obs))
        for upgrade in self.tech_list:
            new_obs.extend(self.upgrade_progress(upgrade, obs))
        self.update_morph_history(obs)
        new_obs = np.array(new_obs, dtype=self.dtype)
        return [new_obs] if self.override else list(obs_pre) + [new_obs]


class ZergUnitProgObsInt(AppendObsInt):
    def reset(self, obs, **kwargs):
        super(ZergUnitProgObsInt, self).reset(obs, **kwargs)
        self.wrapper = ZergUnitProg(self.unwrapped().dc.sd.TT,
                                    override=self.override,
                                    space_old=self.inter.observation_space)
