from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

import os

def launch_setup(context, *args, **kwargs):
    # if not 'TURTLEBOT3_MODEL' in os.environ:
    #     os.environ['TURTLEBOT3_MODEL'] = 'waffle'
    # ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Directories

    dataset_path = LaunchConfiguration('dataset_path').perform(context)
    rate = float(LaunchConfiguration('rate').perform(context))
    dataset_player_node = Node(
        package='ratsim_ros2',
        executable='play_ratsim_bag', 
        name='play_ratsim_bag',
        parameters=[{
            'dataset_path': dataset_path,
            'rate': rate,
            'use_sim_time': True
        }]
    )

    return [
        dataset_player_node,
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('dataset_path', description='Path to dataset.pickle'),

        DeclareLaunchArgument('rate', default_value='1.0', description='Playback rate'),

        OpaqueFunction(function=launch_setup)
    ])
