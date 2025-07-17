#!/usr/bin/env python3

from ratsim.nav import navsim
from ratsim.nav.navsim import NavSim
from ratsim.roslike_unity_connector.connector import *
from ratsim.roslike_unity_connector.message_definitions import *
from ratsim.nav.reactive_controller import *
from ratsim.roslike_unity_connector.bag import MessageBag

from ratsim_ros2.conversions import *
from rosgraph_msgs.msg import Clock
import sys

import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        save_filename = sys.argv[1]
        rate = float(sys.argv[2]) if len(sys.argv) == 3 else 1
        self.get_logger().info('Opening ratbag file: ' + save_filename )

        # Add ROS2 publishers for Odometry and LaserScan and time
        pub_odom = self.create_publisher(Odometry, '/odom', 10)
        pub_lidar = self.create_publisher(LaserScan, '/scan', 10)
        pub_time = self.create_publisher(Clock, '/clock', 1)

        deltatime = 0.1
        step_idx = 0

        bag = MessageBag(save_filename)
        
        print("num steps:", len(bag.steps))
        rate = rate / deltatime
        walltime_target_dt = 1.0 / rate
        
        for i, step in enumerate(bag.steps):
            stepstarttime = time.time()
            simtime = i * deltatime

            lidarmsg = step["/lidar2d"][0]
            gt_pose_msg = step["/rat1_pose"][0]

            # Convert Lidar2DMessage to LaserScan
            ros_odom_msg = convert_twist2d_to_odometry(gt_pose_msg, simtime)
            # Convert odom to ROS odom msg
            ros_lidar_msg = convert_lidar2d_to_laserscan(lidarmsg, simtime)
            # Publish ROS time 
            clock_msg = Clock()
            clock_msg.clock = create_ros_time(simtime)
            pub_time.publish(clock_msg)

            # Publish the messages
            pub_odom.publish(ros_odom_msg)
            pub_lidar.publish(ros_lidar_msg)

            walltime_elapsed = time.time() - stepstarttime
            # Wait for the remaining time to match the target rate
            if walltime_elapsed < walltime_target_dt:
                time.sleep(walltime_target_dt - walltime_elapsed)

            print(f"Published step {i+1}/{len(bag.steps)}: simtime={simtime:.2f}s")

        print("Finished publishing all steps.")

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
