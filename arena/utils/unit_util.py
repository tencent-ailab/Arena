from pysc2.lib.typeenums import UNIT_TYPEID
from pysc2.lib.typeenums import UPGRADE_ID
from arena.utils.constant import AllianceType
import numpy as np


def collect_units_by_type(units, unit_type, alliance=AllianceType.SELF.value):
    """ return unit's ID in the same type """
    return [u for u in units
            if u.unit_type == unit_type and u.alliance == alliance]


def collect_units_by_types(units, unit_types):
    """ return unit's ID in the unit_types list """
    return [u for u in units if u.unit_type in unit_types]


def collect_units_by_alliance(units, alliance=AllianceType.SELF.value):
    return [u for u in units if u.alliance == alliance]


def find_units_by_tag(units, tag):
    return [u for u in units if u.tag == tag]


def find_weakest(units):
    """ find the weakest one to 'unit' within the list 'units' """
    if not units:
        return None
    dd = np.asarray([u.health for u in units])
    return units[dd.argmin()]


def find_strongest(units):
    """ find the strongest one to 'unit' within the list 'units' """
    if not units:
        return None
    dd = np.asarray([u.health for u in units])
    return units[dd.argmax()]


def merge_units_from(l1, l2):
    """ Merge info from l2 to l1 """
    for u2 in l2:
        matched_u = [u for u in l1 if u.tag == u2.tag]
        if len(matched_u) > 0: # Found
            assert len(matched_u) == 1
            u1 = matched_u[0]
            # Set each unset field
            if not u1.HasField('weapon_cooldown'):
                u1.weapon_cooldown = u2.weapon_cooldown
            if not u1.HasField('engaged_target_tag'):
                u1.engaged_target_tag = u2.engaged_target_tag
            for od2 in u2.orders:
                if not od2 in u1.orders:
                    u1.orders.extend([od2])
        else: # Not Found
            l1.extend([u2])
    return


def merge_units(l1, l2):
    " Merge units info from two raw_data.units to complete info"
    merge_units_from(l1, l2)
    merge_units_from(l2, l1)

