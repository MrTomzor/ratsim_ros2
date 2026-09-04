"""Launch file for frontier-based foraging exploration.

Starts the Unity-ROS2 bridge and the foraging explorer node.

Usage:
    ros2 launch ratsim_ros2 frontier_exploration.launch.py

    # With config overrides:
    ros2 launch ratsim_ros2 frontier_exploration.launch.py \
        world_config_json:='{"world_bounds/width": 500, "seed": 42}' \
        agent_config_json:='{"prefab_name": "SphereAgent", ...}' \
        seeds:='1,2,3'

    # Lockstep (default): the sim waits for each command — deterministic,
    # RL-style stepping; replan_interval counts sim seconds. Free-running
    # mode (wall-clock timers, non-deterministic):
    ros2 launch ratsim_ros2 frontier_exploration.launch.py lockstep:=false
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
    lockstep = LaunchConfiguration("lockstep").perform(context).lower() in ("true", "1")

    def farg(name: str) -> float:
        return float(LaunchConfiguration(name).perform(context))

    def iarg(name: str) -> int:
        return int(LaunchConfiguration(name).perform(context))

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
                "lockstep": lockstep,
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
                "grid_resolution": farg("grid_resolution"),
                "inflation_radius": farg("inflation_radius"),
                "reward_descriptor_index": iarg("reward_descriptor_index"),
                "descriptor_dimension": iarg("descriptor_dimension"),
                "max_linear_vel": farg("max_linear_vel"),
                "max_angular_vel": farg("max_angular_vel"),
                "lookahead_dist": farg("lookahead_dist"),
                "min_lookahead": farg("min_lookahead"),
                "goal_reached_dist": farg("goal_reached_dist"),
                "path_clearance": farg("path_clearance"),
                "path_clearance_weight": farg("path_clearance_weight"),
                "curve_speed_margin": farg("curve_speed_margin"),
                "astar_max_expansions": iarg("astar_max_expansions"),
                "frontier_min_size": iarg("frontier_min_size"),
                "obstacle_slowdown_dist": farg("obstacle_slowdown_dist"),
                "safety_dist": farg("safety_dist"),
                "pure_rotation_threshold": farg("pure_rotation_threshold"),
                "reward_lost_ticks": iarg("reward_lost_ticks"),
                "reward_approach_dist": farg("reward_approach_dist"),
                "goal_switch_margin": farg("goal_switch_margin"),
                "goal_selection": LaunchConfiguration("goal_selection").perform(context),
                "replan_interval": farg("replan_interval"),
                "control_rate": farg("control_rate"),
                "map_publish_interval": farg("map_publish_interval"),
                "lockstep": lockstep,
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
            DeclareLaunchArgument(
                "task_preset",
                default_value="volumetric_exploration_5000_collision_penalty",
            ),
            DeclareLaunchArgument("scene_name", default_value="Wildfire"),
            DeclareLaunchArgument("seeds", default_value="1"),
            DeclareLaunchArgument("episodes_per_seed", default_value="1"),
            # 0 = unthrottled; pass rtf:=1.0 to watch in real time
            DeclareLaunchArgument("rtf", default_value="0.0"),
            DeclareLaunchArgument("lockstep", default_value="true"),
            # Explorer parameters (defaults match the node's declare_parameter defaults)
            DeclareLaunchArgument("grid_resolution", default_value="0.3"),
            DeclareLaunchArgument("inflation_radius", default_value="0.4"),
            # 0 = reward_obj1 in the default agent preset's
            # reward_and_boundary_only semantic set
            DeclareLaunchArgument("reward_descriptor_index", default_value="0"),
            DeclareLaunchArgument("descriptor_dimension", default_value="3"),
            DeclareLaunchArgument("max_linear_vel", default_value="10.0"),
            DeclareLaunchArgument("max_angular_vel", default_value="1.5"),
            DeclareLaunchArgument("lookahead_dist", default_value="5.0"),
            DeclareLaunchArgument("min_lookahead", default_value="0.8"),
            DeclareLaunchArgument("goal_reached_dist", default_value="1.0"),
            DeclareLaunchArgument("path_clearance", default_value="1.0"),
            DeclareLaunchArgument("path_clearance_weight", default_value="3.0"),
            DeclareLaunchArgument("curve_speed_margin", default_value="0.9"),
            DeclareLaunchArgument("astar_max_expansions", default_value="60000"),
            DeclareLaunchArgument("frontier_min_size", default_value="5"),
            DeclareLaunchArgument("obstacle_slowdown_dist", default_value="1.5"),
            DeclareLaunchArgument("safety_dist", default_value="1.5"),
            DeclareLaunchArgument("pure_rotation_threshold", default_value="1.2"),
            DeclareLaunchArgument("reward_lost_ticks", default_value="3"),
            DeclareLaunchArgument("reward_approach_dist", default_value="2.0"),
            DeclareLaunchArgument("goal_switch_margin", default_value="0.7"),
            # "floodfill" = Dijkstra flood, cheapest path cost wins;
            # "euclidean" = straight-line ranking + per-candidate A*
            DeclareLaunchArgument("goal_selection", default_value="floodfill"),
            DeclareLaunchArgument("replan_interval", default_value="2.0"),
            DeclareLaunchArgument("control_rate", default_value="50.0"),
            DeclareLaunchArgument("map_publish_interval", default_value="0.5"),
            OpaqueFunction(function=launch_setup),
        ]
    )
