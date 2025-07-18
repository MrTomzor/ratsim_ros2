from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

import os

def launch_setup(context, *args, **kwargs):
    # Directories
    pkg_nav2_bringup = get_package_share_directory(
        'nav2_bringup')
    pkg_rtabmap_demos = get_package_share_directory(
        'rtabmap_demos')
    
    world_name = LaunchConfiguration('world').perform(context)
    
    icp_odometry = LaunchConfiguration('icp_odometry').perform(context)
    icp_odometry = icp_odometry == 'True' or icp_odometry == 'true'
    if icp_odometry:
        # modified nav2 params to use icp_odom instead odom frame
        nav2_params_file = PathJoinSubstitution(
            [FindPackageShare('rtabmap_demos'), 'params', 'turtlebot3_scan_nav2_params.yaml']
        )
    else:
        # original nav2 params
        nav2_params_file = PathJoinSubstitution(
            [FindPackageShare('nav2_bringup'), 'params', 'nav2_params.yaml']
        )

    # Paths
    nav2_launch = PathJoinSubstitution(
        [pkg_nav2_bringup, 'launch', 'navigation_launch.py'])
    rviz_launch = PathJoinSubstitution(
        [pkg_nav2_bringup, 'launch', 'rviz_launch.py'])
    rtabmap_launch = PathJoinSubstitution(
        [pkg_rtabmap_demos, 'launch', 'turtlebot3', 'turtlebot3_scan.launch.py'])

    # To use ICP odometry, we should increase clock rate of gazebo, we copied content of
    # turtlebot3_gazebo/launch/turtlebot3_world.launch here
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    # pkg_gazebo_ros = get_package_share_directory('gazebo_ros')


    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as clock_override_file:
        clock_override_file.write("---\n"+
                  "gazebo:\n"+
                  "    ros__parameters:\n"+
                  "        publish_rate: 100.0")

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': LaunchConfiguration('x_pose'),
            'y_pose': LaunchConfiguration('y_pose')
        }.items()
    )
    
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav2_launch]),
        launch_arguments=[
            ('use_sim_time', 'true'),
            ('params_file', nav2_params_file)
        ]
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch])
    )
    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch]),
        launch_arguments=[
            ('localization', LaunchConfiguration('localization')),
            ('use_sim_time', 'true')
        ]
    )


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
        # Nodes to launch
        nav2,
        rviz,
        rtabmap,
        dataset_player_node,
        # gzserver_cmd,
        # gzclient_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('dataset_path', description='Path to dataset.pickle'),

        DeclareLaunchArgument('rate', default_value='1.0', description='Playback rate'),

        # Launch arguments
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Launch in localization mode.'),
        
        DeclareLaunchArgument(
            'world', default_value='world',
            choices=['world', 'house', 'dqn_stage1', 'dqn_stage2', 'dqn_stage3', 'dqn_stage4'],
            description='Turtlebot3 gazebo world.'),
        
        DeclareLaunchArgument(
            'icp_odometry', default_value='false',
            description='Launch ICP odometry on top of wheel odometry.'),
        
        DeclareLaunchArgument(
            'x_pose', default_value='-2.0',
            description='Initial position of the robot in the simulator.'),
        
        DeclareLaunchArgument(
            'y_pose', default_value='0.5',
            description='Initial position of the robot in the simulator.'),

        OpaqueFunction(function=launch_setup)
    ])
