#!/usr/bin/env python3
"""Unity-ROS2 bridge node.

Owns the RoslikeUnityConnector, drives the sim loop, and republishes
Unity sensor data as standard ROS2 topics.  Subscribes to /cmd_vel for
action commands.  Manages episode lifecycle (reset, metrics, TaskTracker).

Published topics:
    /scan              (sensor_msgs/LaserScan)
    /odom              (nav_msgs/Odometry)
    /semantic_lidar    (std_msgs/Float32MultiArray)
    /clock             (rosgraph_msgs/Clock)
    /tf                (tf2_msgs/TFMessage via tf2_ros)
    /world_bounds      (std_msgs/Float32MultiArray)  — latched
    /episode_active    (std_msgs/Bool)               — latched

Subscribed topics:
    /cmd_vel           (geometry_msgs/Twist)

Services:
    /reset_episode     (std_srvs/Trigger)
"""

import json
import sys
import math
import threading
import time

import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray, String, Header
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time
import tf2_ros

from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import (
    BoolMessage,
    StringMessage,
    TwistMessage,
)
from ratsim.config_blender import blend_presets, to_entries_json
from ratsim.task_tracker import TaskTracker
from ratsim.transforms import yaw_from_quat

from ratsim_ros2.conversions import (
    convert_lidar2d_to_laserscan,
    convert_pose_to_odometry,
    create_ros_time,
)


class UnityRos2Bridge(Node):
    def __init__(self):
        super().__init__("unity_ros2_bridge")

        # -- Parameters --
        self.declare_parameter("world_config_json", "")
        self.declare_parameter("agent_config_json", "")
        self.declare_parameter("task_config_json", "")
        self.declare_parameter("world_preset", "default")
        self.declare_parameter("agent_preset", "sphereagent_2d_lidar")
        self.declare_parameter("task_preset", "default")
        self.declare_parameter("scene_name", "Wildfire")
        self.declare_parameter("seeds", "1,2,3,4,5,6,7,8,9,10")
        self.declare_parameter("episodes_per_seed", 1)
        self.declare_parameter("episode_max_steps", 2000)

        # Load configs: prefer JSON params, fall back to presets via blend_presets
        world_json_str = self.get_parameter("world_config_json").value or ""
        agent_json_str = self.get_parameter("agent_config_json").value or ""
        task_json_str = self.get_parameter("task_config_json").value or ""

        if world_json_str and world_json_str != "{}":
            self.world_config = json.loads(world_json_str)
        else:
            preset = self.get_parameter("world_preset").value
            self.world_config = blend_presets("world", [preset])
            self.get_logger().info(f"Loaded world preset: {preset}")

        if agent_json_str and agent_json_str != "{}":
            self.agent_config = json.loads(agent_json_str)
        else:
            preset = self.get_parameter("agent_preset").value
            self.agent_config = blend_presets("agents", [preset])
            self.get_logger().info(f"Loaded agent preset: {preset}")

        if task_json_str and task_json_str != "{}":
            self.task_config = json.loads(task_json_str)
        else:
            preset = self.get_parameter("task_preset").value
            self.task_config = blend_presets("task", [preset])
            self.get_logger().info(f"Loaded task preset: {preset}")

        self.scene_name = self.get_parameter("scene_name").value

        seeds_str = self.get_parameter("seeds").value
        self.seeds = [int(s.strip()) for s in seeds_str.split(",")]
        self.episodes_per_seed = self.get_parameter("episodes_per_seed").value
        self.episode_max_steps_param = self.get_parameter("episode_max_steps").value

        # -- TaskTracker --
        self.task_tracker = TaskTracker(self.task_config)

        # -- Unity connector --
        self.get_logger().info("Connecting to Unity...")
        self.conn = RoslikeUnityConnector(verbose=False)
        self.conn.connect()
        self.get_logger().info("Connected to Unity.")

        # Select scene
        self.conn.publish(StringMessage(data=self.scene_name), "/sim_control/scene_select")
        self.conn.send_messages_and_step(enable_physics_step=False)
        self.conn.read_messages_from_unity()

        # Send agent config (always — Unity's AgentLoader needs this to spawn the agent)
        agent_json = to_entries_json(self.agent_config)
        self.conn.publish(StringMessage(data=agent_json), "/sim_control/agent_config")
        self.conn.send_messages_and_step(enable_physics_step=False)
        self.conn.read_messages_from_unity()
        self.get_logger().info(f"Sent agent config: {agent_json[:200]}")

        # -- ROS2 publishers --
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_semantic = self.create_publisher(Float32MultiArray, "/semantic_lidar", 10)

        # Latched (transient local) topics
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub_world_bounds = self.create_publisher(
            Float32MultiArray, "/world_bounds", latched_qos
        )
        self.pub_episode_active = self.create_publisher(Bool, "/episode_active", latched_qos)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # -- ROS2 subscribers --
        self.latest_cmd_vel = TwistMessage(
            linear_x=0.0, linear_y=0.0, linear_z=0.0,
            angular_x=0.0, angular_y=0.0, angular_z=0.0,
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

        # -- ROS2 services --
        self.create_service(Trigger, "/reset_episode", self._reset_episode_cb)

        # -- Episode state machine --
        self.sim_step = 0
        self.episode_active = False
        self._seed_idx = 0
        self._ep_idx = 0
        self._all_done = False
        self._needs_reset = True  # start by resetting the first episode
        self._stop_event = threading.Event()  # for clean shutdown

        # -- Sim loop runs on a dedicated thread so it doesn't starve the
        #    executor's DDS callbacks (subscription delivery, service calls).
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        self.get_logger().info("Bridge ready. Waiting for DDS discovery before first episode...")

    # ------------------------------------------------------------------
    # Sim loop (runs on dedicated thread, NOT on executor)
    # ------------------------------------------------------------------

    def _sim_loop(self):
        """Blocking loop that drives the Unity simulation.

        Runs on its own thread so the ROS2 executor threads stay free to
        process subscription callbacks (/cmd_vel) and service calls.
        """
        # Wait for DDS discovery — executor needs time to set up transports
        self.get_logger().info("Waiting 1s for DDS discovery...")
        time.sleep(1.0)
        self._publish_world_bounds()
        self.get_logger().info("DDS discovery period done. Starting episodes.")

        while rclpy.ok() and not self._all_done:
            # Reset episode if needed
            if self._needs_reset:
                seed = self.seeds[self._seed_idx]
                actual_seed = seed + self._ep_idx * 10000
                self.get_logger().info(
                    f"Starting episode: seed={seed}, ep_idx={self._ep_idx}, actual_seed={actual_seed}"
                )
                self._do_reset_episode(seed=actual_seed)
                self._needs_reset = False
                # Let explorer see /episode_active before first sim tick
                time.sleep(0.1)
                continue

            if not self.episode_active:
                time.sleep(0.01)
                continue

            # Sim tick (blocks on TCP round-trip ~16ms)
            self._sim_tick()

            # Check termination
            terminated = self.task_tracker.is_terminated()
            truncated = self.sim_step >= self.task_tracker.episode_max_steps

            if terminated or truncated:
                self.episode_active = False
                active_msg = Bool()
                active_msg.data = False
                self.pub_episode_active.publish(active_msg)

                seed = self.seeds[self._seed_idx]
                result = {
                    "seed": seed,
                    "episode_idx": self._ep_idx,
                    "steps": self.sim_step,
                    "total_score": self.task_tracker.get_total_score(),
                    "objects_found": self.task_tracker.get_num_reward_objs_picked_up(),
                    "collisions": self.task_tracker.get_collision_count(),
                    "termination_reason": (
                        self.task_tracker.get_termination_reason()
                        if terminated
                        else "max_steps"
                    ),
                    "terminated": terminated,
                    "truncated": truncated,
                }
                # JSON line to stdout for test.py
                print(json.dumps(result), flush=True)
                self.get_logger().info(
                    f"Episode done: seed={seed}, steps={self.sim_step}, "
                    f"objects={result['objects_found']}, "
                    f"score={result['total_score']:.2f}, "
                    f"reason={result['termination_reason']}"
                )

                # Advance to next episode
                self._ep_idx += 1
                if self._ep_idx >= self.episodes_per_seed:
                    self._ep_idx = 0
                    self._seed_idx += 1
                    if self._seed_idx >= len(self.seeds):
                        self.get_logger().info("All episodes complete.")
                        self._all_done = True
                        return
                self._needs_reset = True

    # ------------------------------------------------------------------
    # World bounds
    # ------------------------------------------------------------------

    def _publish_world_bounds(self):
        w = float(self.world_config.get("world_bounds/width", 2000))
        h = float(self.world_config.get("world_bounds/height", 2000))
        msg = Float32MultiArray()
        msg.data = [w, h]
        self.pub_world_bounds.publish(msg)
        self.get_logger().info(f"Published world bounds: {w} x {h}")

    # ------------------------------------------------------------------
    # Cmd vel callback
    # ------------------------------------------------------------------

    def _cmd_vel_cb(self, msg: Twist):
        if not hasattr(self, '_cmd_vel_log_count'):
            self._cmd_vel_log_count = 0
        self._cmd_vel_log_count += 1
        if self._cmd_vel_log_count <= 3 or (msg.linear.x != 0.0 and self._cmd_vel_log_count % 100 == 0):
            self.get_logger().info(
                f"Received cmd_vel: lin_x={msg.linear.x:.3f}, ang_z={msg.angular.z:.3f}"
            )
        self.latest_cmd_vel = TwistMessage(
            linear_x=msg.linear.x,
            linear_y=msg.linear.y,
            linear_z=msg.linear.z,
            angular_x=msg.angular.x,
            angular_y=msg.angular.y,
            angular_z=msg.angular.z,
        )

    # ------------------------------------------------------------------
    # Reset episode service
    # ------------------------------------------------------------------

    def _reset_episode_cb(self, request, response):
        self.get_logger().info("Reset episode requested via service.")
        self._do_reset_episode()
        response.success = True
        response.message = "Episode reset."
        return response

    def _do_reset_episode(self, seed: int | None = None):
        """Send world config + reset to Unity.  Resets TaskTracker."""
        self.episode_active = False
        active_msg = Bool()
        active_msg.data = False
        self.pub_episode_active.publish(active_msg)

        cfg = dict(self.world_config)
        if seed is not None:
            cfg["seed"] = seed

        config_json = to_entries_json(cfg)
        self.conn.publish(StringMessage(data=config_json), "/sim_control/world_config")
        self.conn.publish(BoolMessage(data=True), "/sim_control/reset_episode")
        self.conn.send_messages_and_step(enable_physics_step=True)
        self.conn.read_messages_from_unity()

        # Extra step to let worldgen settle
        self.conn.send_messages_and_step(enable_physics_step=True)
        self.conn.read_messages_from_unity()

        self.task_tracker.reset()
        self.sim_step = 0

        # Reset cmd_vel to zero
        self.latest_cmd_vel = TwistMessage(
            linear_x=0.0, linear_y=0.0, linear_z=0.0,
            angular_x=0.0, angular_y=0.0, angular_z=0.0,
        )

        # Re-publish world bounds in case explorer joined late
        self._publish_world_bounds()

        self.episode_active = True
        active_msg.data = True
        self.pub_episode_active.publish(active_msg)

    # ------------------------------------------------------------------
    # Sim tick
    # ------------------------------------------------------------------

    def _sim_tick(self):
        """One step: send cmd_vel, step Unity, read messages, publish ROS2."""
        # Send action to Unity
        self.conn.publish(self.latest_cmd_vel, "/cmd_vel")
        self.conn.send_messages_and_step(enable_physics_step=True)
        msgs = self.conn.read_messages_from_unity()

        if isinstance(msgs, int):
            # Timeout
            self.get_logger().warn("Unity timeout during sim tick.")
            return

        self.sim_step += 1
        sim_time = self.sim_step * 0.02  # 50Hz physics

        # Update task tracker
        self.task_tracker.update_with_unity_msgs(msgs)

        # Publish clock
        clock_msg = Clock()
        clock_msg.clock = create_ros_time(sim_time)
        self.pub_clock.publish(clock_msg)

        # Publish scan
        if "/lidar2d" in msgs:
            lidar_msg = msgs["/lidar2d"][0]
            ros_scan = convert_lidar2d_to_laserscan(lidar_msg, sim_time)
            self.pub_scan.publish(ros_scan)

            # Publish semantic lidar (raw descriptors)
            if lidar_msg.descriptors is not None:
                sem_msg = Float32MultiArray()
                sem_msg.data = [float(d) for d in lidar_msg.descriptors]
                self.pub_semantic.publish(sem_msg)

        # Publish odom + tf (try multiple pose topic names)
        pose_msg = None
        for topic in ["/rat1_pose", "/rat1_pose_from_start"]:
            if topic in msgs:
                pose_msg = msgs[topic][0]
                break

        if pose_msg is not None:
            ros_odom = convert_pose_to_odometry(pose_msg, sim_time)
            self.pub_odom.publish(ros_odom)

            # Broadcast TF: odom -> base_link
            t = TransformStamped()
            t.header.stamp = create_ros_time(sim_time)
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"
            t.transform.translation.x = pose_msg.x if pose_msg.x else 0.0
            t.transform.translation.y = pose_msg.y if pose_msg.y else 0.0
            t.transform.translation.z = pose_msg.z if pose_msg.z else 0.0
            t.transform.rotation.x = pose_msg.qx if pose_msg.qx else 0.0
            t.transform.rotation.y = pose_msg.qy if pose_msg.qy else 0.0
            t.transform.rotation.z = pose_msg.qz if pose_msg.qz else 0.0
            t.transform.rotation.w = pose_msg.qw if pose_msg.qw else 1.0
            self.tf_broadcaster.sendTransform(t)

        # Log periodically
        if self.sim_step % 100 == 0:
            topics = list(msgs.keys())
            self.get_logger().info(
                f"Step {self.sim_step}, score={self.task_tracker.get_total_score():.2f}, "
                f"objects={self.task_tracker.get_num_reward_objs_picked_up()}, "
                f"topics={topics}"
            )
        if self.sim_step == 1:
            self.get_logger().info(f"First sim tick, Unity topics: {list(msgs.keys())}")


def main(args=None):
    rclpy.init(args=args)
    node = UnityRos2Bridge()
    # Use MultiThreadedExecutor so the blocking sim-tick timer doesn't
    # starve DDS discovery, transport flushing, or subscription callbacks.
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
