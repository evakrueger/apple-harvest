#!/usr/bin/env python3
# Starts only the RealSense cameras enabled in harvest/config/cameras.yaml
# (launch_vision.launch.py starts these plus the other cameras and vision nodes).
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from harvest import cameras


def launch_setup(context):
    config = cameras.load_config(LaunchConfiguration("cameras_config").perform(context))
    return cameras.camera_nodes(config, camera_type='realsense')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("cameras_config", default_value=cameras.default_config_path(),
                              description="Camera config file (serials, which cameras are enabled)."),
        OpaqueFunction(function=launch_setup),
    ])
