from ratsim.roslike_unity_connector.connector import *
from ratsim.roslike_unity_connector.message_definitions import *
import numpy as np
from ratsim_ros2.conversions import *
import copy
# TODO - fix relative imports

import rclpy
from rclpy.node import Node

class Ros2RatsimAgentConnector(Node):
    def __init__(self):
        super().__init__('my_node')
        # TODO listen for twist msg
        self.action_twist_topic = "/cmd_vel"
        self.twist_subscriber = self.create_subscription(
            Twist, self.action_twist_topic, self.twist_callback, 10)

        self.latest_action_navsim_msgs = {}
        self.lpublishers = {}
        self.wait_for_action = False
        pass

    def twist_callback(self, msg):
        self.latest_action_navsim_msgs = {self.action_twist_topic: [TwistMessage(
            linear_x=msg.linear.x,
            linear_y=msg.linear.y,
            linear_z=msg.linear.z,
            angular_x=msg.angular.x,
            angular_y=msg.angular.y,
            angular_z=msg.angular.z
        )]}
        pass

    def step(self, input_navsim_msgs, time):
        topics = copy.deepcopy(input_navsim_msgs.keys())
        for topic in topics:
            ros_msg = convert_msg_from_ratsim_to_ros(input_navsim_msgs[topic][0], time)
            if not topic in self.publishers:
                newpub = self.create_publisher(ros_msg.__class__, topic, 10)
                self.lpublishers[topic] = newpub
            self.lpublishers[topic].publish(ros_msg)

        # IF ENABLED - WAIT FOR SOME ACTION
        if self.wait_for_action:
            # TODO - wait for action
            # self.get_logger().info("Waiting for action")
            # rclpy.spin_once(self)
            return {}

        out_msgs = self.latest_action_navsim_msgs
        return out_msgs


