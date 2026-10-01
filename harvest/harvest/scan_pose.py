"""
Arm poses saved from the GUI's Preferences tab (set to wherever the arm was freedriven):
  'scan'  where the vision approach scans for apples; read by start_harvest.py. Without it, the built-in scan pose
          in move_arm.cpp (move_arm_to_scan) is used.
  'home'  the home pose; the GUI sets it on move_arm (its home_joint_positions parameter) before each pick. Without
          it, move_arm's built-in home is used.

Kept in harvest/config/<name>_pose.yaml in the source tree (found through this file's real path, which the
symlink install points back to), so they survive rebuilds.
"""

import time
from pathlib import Path

import yaml

# move group order, which move_arm_to_joints expects
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

PATH = Path(__file__).resolve().parents[1] / 'config' / 'scan_pose.yaml'  # the scan pose


def path(name='scan'):
    return PATH if name == 'scan' else PATH.with_name(f'{name}_pose.yaml')


def load(name='scan'):
    """{'positions': [rad, in JOINT_NAMES order], 'saved': 'YYYY-mm-dd HH:MM:SS'}, or None if not set."""
    try:
        with open(path(name)) as f:
            data = yaml.safe_load(f)
        by_name = dict(zip(data['joint_names'], data['positions']))
        return {'positions': [float(by_name[joint]) for joint in JOINT_NAMES], 'saved': str(data.get('saved', ''))}
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def save(positions_by_name, name='scan'):
    """Save joint positions (a {joint name: rad} dict, e.g. from /joint_states) as the named pose."""
    data = {'joint_names': JOINT_NAMES,
            'positions': [round(float(positions_by_name[joint]), 5) for joint in JOINT_NAMES],
            'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(path(name), 'w') as f:
        f.write(f'# {name.capitalize()} pose, set from the harvest GUI (Preferences > Set {name} pose to current arm\n'
                f'# position). Joint positions in radians. Delete this file to use the {name} pose built into move_arm.cpp.\n')
        yaml.safe_dump(data, f, sort_keys=False)
    return load(name)


def clear(name='scan'):
    """Go back to the built-in pose."""
    try:
        path(name).unlink()
    except FileNotFoundError:
        pass
