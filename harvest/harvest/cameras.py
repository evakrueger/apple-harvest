"""
Camera configuration shared by the vision launch files and start_harvest.py.

Cameras are described in config/cameras.yaml; this module turns that into launch actions and
the list of image topics to record, so topic names are defined in one place.
"""

import importlib.util
import os

import yaml
from ament_index_python.packages import get_package_share_directory

# realsense node topic (relative to /<camera_namespace>/<name>/) -> published as /<name>_<suffix>
REALSENSE_TOPICS = {
    'color/image_raw': 'image_raw',
    'color/camera_info': 'camera_info',
    'aligned_depth_to_color/image_raw': 'aligned_depth_image_raw',
    'aligned_depth_to_color/camera_info': 'aligned_depth_camera_info',
    'depth/image_rect_raw': 'depth_image_raw',
    'depth/camera_info': 'depth_camera_info',
    'depth/color/points': 'points',
}
REALSENSE_NAMESPACE = 'camera'

# topic the gripper_palm_camera node (and visual_servo) use internally
PALM_CAMERA_NODE_TOPIC = 'gripper/rgb_palm_camera/image_raw'


def default_config_path():
    return os.path.join(get_package_share_directory('harvest'), 'config', 'cameras.yaml')


def load_config(path=None):
    with open(path or default_config_path(), 'r') as f:
        config = yaml.safe_load(f)
    _validate(config)
    return config


def enabled_cameras(config, camera_type=None):
    return {name: cam for name, cam in config['cameras'].items()
            if cam.get('enabled', False) and (camera_type is None or cam['type'] == camera_type)}


def topic(name, suffix='image_raw'):
    return f'/{name}_{suffix}'


def recorded_image_topics(config):
    """Image topics start_harvest records: /<name>_image_raw for each enabled camera with record: true."""
    return [topic(name) for name, cam in enabled_cameras(config).items() if cam.get('record', False)]


def _validate(config):
    serials = {}
    for name, cam in enabled_cameras(config, 'realsense').items():
        serial = str(cam.get('serial', '') or '')
        if not serial:
            raise ValueError(f"cameras.yaml: enabled realsense camera '{name}' has no serial")
        if serial in serials:
            raise ValueError(f"cameras.yaml: realsense cameras '{serials[serial]}' and '{name}' share serial {serial}")
        serials[serial] = name


# --- launch actions ---

def _rs_launch_default_parameters():
    """The parameter set realsense2_camera's rs_launch.py passes to the node (all as strings)."""
    path = os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
    spec = importlib.util.spec_from_file_location('rs_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {p['name']: p['default'] for p in module.configurable_parameters}


def realsense_node(name, cam):
    # Same node and parameters rs_launch.py would create, but launched directly: including rs_launch
    # made it warn about every launch argument of the parent launch file, and a Node lets us remap
    # its topics to /<name>_*.
    from launch.substitutions import TextSubstitution
    from launch_ros.actions import Node
    params = _rs_launch_default_parameters()
    params.update({
        'camera_name': name,
        'camera_namespace': REALSENSE_NAMESPACE,
        'serial_no': f"'{cam['serial']}'",
        'enable_color': 'true',
        'enable_depth': 'true',
        'align_depth.enable': str(cam.get('align_depth', True)).lower(),
        'publish_tf': str(cam.get('publish_tf', False)).lower(),
        'pointcloud.enable': str(cam.get('pointcloud', True)).lower(),
    })
    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace=REALSENSE_NAMESPACE,
        name=name,
        # as substitutions (like rs_launch's LaunchConfigurations) so values are YAML-parsed into their types
        parameters=[{k: TextSubstitution(text=v) for k, v in params.items()}],
        remappings=[(f'/{REALSENSE_NAMESPACE}/{name}/{src}', topic(name, dst)) for src, dst in REALSENSE_TOPICS.items()],
        output='screen',
        arguments=['--ros-args', '--log-level', params['log_level']],
        emulate_tty=True,
    )


def usb_cam_node(name, cam):
    from launch_ros.actions import Node
    return Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name=f'{name}_node',
        # resolve symlinks here (by-id / udev names); usb_cam mangles relative symlinks into /dev/../../videoN
        parameters=[{'video_device': os.path.realpath(cam['device']),
                     'camera_name': name,
                     'frame_id': name}],
        remappings=[('image_raw', topic(name)), ('camera_info', topic(name, 'camera_info'))],
        output='screen',
    )


def palm_camera_node(name, cam):
    from launch_ros.actions import Node
    return Node(
        package='robot_custom_hardware',
        executable='gripper_palm_camera',
        name='gripper_palm_camera',
        # device: a path (by-id / udev name; follows the camera to any port) or N of /dev/videoN
        parameters=[{'palm_camera_device': str(cam['device'])} if isinstance(cam['device'], str)
                    else {'palm_camera_device_num': int(cam['device'])}],
        remappings=[(PALM_CAMERA_NODE_TOPIC, topic(name))],
    )


CAMERA_NODE_BUILDERS = {
    'realsense': realsense_node,
    'usb_cam': usb_cam_node,
    'palm_camera': palm_camera_node,
}


def camera_nodes(config, camera_type=None):
    return [CAMERA_NODE_BUILDERS[cam['type']](name, cam) for name, cam in enabled_cameras(config, camera_type).items()]
