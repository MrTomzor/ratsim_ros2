#!/usr/bin/env python3

from ratsim.nav import navsim
from ratsim.nav.navsim import NavSim
from ratsim.roslike_unity_connector.connector import *
from ratsim.roslike_unity_connector.message_definitions import *
from ratsim.nav.reactive_controller import *

from ratsim_ros2.conversions import *

import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Hello from my_node!')


        # Add ROS2 publishers for Odometry and LaserScan
        pub_odom = self.create_publisher(Odometry, '/odom', 10)
        pub_lidar = self.create_publisher(LaserScan, '/scan', 10)

        sim = NavSim()
        
        # First step
        last_obsv, done = sim.step()

        reactive_controller = ReactiveController(2, 4, 1, 0.5, dist_threshold1=3, dist_threshold2=5, ignore_colored=True) 

        deltatime = 0.1
        step_idx = 0

        
        while True:
            simtime = step_idx * deltatime

            lidarmsg = last_obsv["/lidar2d"][0]
            gt_pose_msg = last_obsv["/rat1_pose"][0]

            # Convert Lidar2DMessage to LaserScan
            ros_odom_msg = convert_twist2d_to_odometry(gt_pose_msg, simtime)
            # Convert odom to ROS odom msg
            ros_lidar_msg = convert_lidar2d_to_laserscan(lidarmsg, simtime)

            # Publish the messages
            pub_odom.publish(ros_odom_msg)
            pub_lidar.publish(ros_lidar_msg)

            # Get reac controller output and apply
            twistmsg = reactive_controller.step(last_obsv)
            last_obsv, done = sim.step({"/cmd_vel": [twistmsg]})

            print("Publishing twist message:", twistmsg.forward, twistmsg.left, twistmsg.radiansCounterClockwise)
            sim.conn.log_connection_stats()

            step_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
