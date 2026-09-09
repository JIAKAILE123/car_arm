#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import ReplaceString


def generate_launch_description():
    navigation2_dir = get_package_share_directory('bot_navigation')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    map_yaml_file = "mapn3.yaml"

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_yaml_path = LaunchConfiguration(
        'map', default=os.path.join(navigation2_dir, f'maps/{map_yaml_file}')
    )
    raw_params = os.path.join(navigation2_dir, 'param', 'originbot_nav2_2.yaml')
    default_bt_xml = os.path.join(
        navigation2_dir, 'behavior_trees', 'navigate_stable.xml'
    )
    # Inject stable BT path into params (replan only if path invalid)
    nav2_param_path = ReplaceString(
        source_file=raw_params,
        replacements={'<nav_bt_xml>': default_bt_xml},
    )
    slam = LaunchConfiguration('slam', default='False')
    rviz_config_dir = os.path.join(nav2_bringup_dir, 'rviz', 'nav2_default_view.rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='Use simulation (Gazebo) clock if true',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=raw_params,
            description='Full path to param file to load',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=map_yaml_path,
            description='Full path to map file to load',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [nav2_bringup_dir, '/launch', '/bringup_launch.py']
            ),
            launch_arguments={
                'map': map_yaml_path,
                'use_sim_time': use_sim_time,
                'params_file': nav2_param_path,
                'slam': slam,
            }.items(),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
    ])
