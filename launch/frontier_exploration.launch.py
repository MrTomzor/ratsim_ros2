"""Launch file for frontier-based foraging exploration.

Starts the Unity-ROS2 bridge and the foraging explorer node.

Usage:
    ros2 launch ratsim_ros2 frontier_exploration.launch.py

    # With config overrides:
    ros2 launch ratsim_ros2 frontier_exploration.launch.py \
        world_config_json:='{"world_bounds/width": 500, "seed": 42}' \
        agent_config_json:='{"prefab_name": "SphereAgent", ...}' \
        seeds:='1,2,3'
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    world_config_json = LaunchConfiguration("world_config_json").perform(context)
    agent_config_json = LaunchConfiguration("agent_config_json").perform(context)
    task_config_json = LaunchConfiguration("task_config_json").perform(context)
    world_preset = LaunchConfiguration("world_preset").perform(context)
    agent_preset = LaunchConfiguration("agent_preset").perform(context)
    task_preset = LaunchConfiguration("task_preset").perform(context)
    scene_name = LaunchConfiguration("scene_name").perform(context)
    seeds = LaunchConfiguration("seeds").perform(context)
    episodes_per_seed = LaunchConfiguration("episodes_per_seed").perform(context)
    rtf = LaunchConfiguration("rtf").perform(context)

    grid_resolution = LaunchConfiguration("grid_resolution").perform(context)
    inflation_radius = LaunchConfiguration("inflation_radius").perform(context)
    reward_descriptor_index = LaunchConfiguration("reward_descriptor_index").perform(context)
    descriptor_dimension = LaunchConfiguration("descriptor_dimension").perform(context)
    max_linear_vel = LaunchConfiguration("max_linear_vel").perform(context)
    max_angular_vel = LaunchConfiguration("max_angular_vel").perform(context)
    lookahead_dist = LaunchConfiguration("lookahead_dist").perform(context)

    bridge_node = Node(
        package="ratsim_ros2",
        executable="unity_ros2_bridge",
        name="unity_ros2_bridge",
        output="screen",
        parameters=[
            {
                "world_config_json": world_config_json,
                "agent_config_json": agent_config_json,
                "task_config_json": task_config_json,
                "world_preset": world_preset,
                "agent_preset": agent_preset,
                "task_preset": task_preset,
                "scene_name": scene_name,
                "seeds": seeds,
                "episodes_per_seed": int(episodes_per_seed),
                "rtf": float(rtf),
            }
        ],
    )

    explorer_node = Node(
        package="ratsim_ros2",
        executable="foraging_explorer",
        name="foraging_explorer",
        output="screen",
        parameters=[
            {
                "grid_resolution": float(grid_resolution),
                "inflation_radius": float(inflation_radius),
                "reward_descriptor_index": int(reward_descriptor_index),
                "descriptor_dimension": int(descriptor_dimension),
                "max_linear_vel": float(max_linear_vel),
                "max_angular_vel": float(max_angular_vel),
                "lookahead_dist": float(lookahead_dist),
            }
        ],
    )

    return [bridge_node, explorer_node]


def generate_launch_description():
    return LaunchDescription(
        [
            # Bridge parameters — pass JSON to override, or use preset names
            DeclareLaunchArgument("world_config_json", default_value=""),
            DeclareLaunchArgument("agent_config_json", default_value=""),
            DeclareLaunchArgument("task_config_json", default_value=""),
            DeclareLaunchArgument("world_preset", default_value="default"),
            DeclareLaunchArgument("agent_preset", default_value="sphereagent_2d_lidar"),
            DeclareLaunchArgument("task_preset", default_value="default"),
            DeclareLaunchArgument("scene_name", default_value="Wildfire"),
            DeclareLaunchArgument("seeds", default_value="1,2,3,4,5,6,7,8,9,10"),
            DeclareLaunchArgument("episodes_per_seed", default_value="1"),
            DeclareLaunchArgument("rtf", default_value="1.0"),
            # Explorer parameters
            DeclareLaunchArgument("grid_resolution", default_value="1.0"),
            DeclareLaunchArgument("inflation_radius", default_value="2.0"),
            DeclareLaunchArgument("reward_descriptor_index", default_value="2"),
            DeclareLaunchArgument("descriptor_dimension", default_value="3"),
            DeclareLaunchArgument("max_linear_vel", default_value="10.0"),
            DeclareLaunchArgument("max_angular_vel", default_value="2.0"),
            DeclareLaunchArgument("lookahead_dist", default_value="5.0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
