#!/usr/bin/env python3
"""Foraging exploration node.

Pure ROS2 node (no ratsim dependency) that builds an occupancy map from
lidar, detects frontiers, plans paths with A*, and follows them.  When a
reward object is detected via semantic lidar descriptors, the agent switches
to COLLECT mode and approaches the reward.

Subscribed topics:
    /scan              (sensor_msgs/LaserScan)
    /odom              (nav_msgs/Odometry)
    /semantic_lidar    (std_msgs/Float32MultiArray)
    /world_bounds      (std_msgs/Float32MultiArray)  — latched
    /episode_active    (std_msgs/Bool)               — latched

Published topics:
    /cmd_vel           (geometry_msgs/Twist)
    /map               (nav_msgs/OccupancyGrid)
    /plan              (nav_msgs/Path)
    /frontiers         (visualization_msgs/MarkerArray)
    /goal_marker       (visualization_msgs/Marker)
"""

import math
import time
from enum import Enum, auto

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from std_msgs.msg import Bool, Float32MultiArray, ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from ratsim_ros2.quadtree import QuadtreeOccupancyGrid


class State(Enum):
    WAITING = auto()   # no world_bounds yet
    EXPLORE = auto()
    COLLECT = auto()


class ForagingExplorer(Node):
    def __init__(self):
        super().__init__("foraging_explorer")

        # -- Parameters --
        self.declare_parameter("grid_resolution", 1.0)
        self.declare_parameter("inflation_radius", 2.0)
        self.declare_parameter("reward_descriptor_index", 2)
        self.declare_parameter("descriptor_dimension", 3)
        self.declare_parameter("max_linear_vel", 10.0)
        self.declare_parameter("max_angular_vel", 1.5)
        self.declare_parameter("collect_approach_dist", 2.0)
        self.declare_parameter("frontier_min_size", 5)
        self.declare_parameter("obstacle_slowdown_dist", 5.0)
        self.declare_parameter("lookahead_dist", 3.0)
        self.declare_parameter("waypoint_reached_dist", 2.0)
        self.declare_parameter("replan_interval", 5.0)  # seconds
        self.declare_parameter("map_publish_interval", 0.5)  # seconds
        self.declare_parameter("safety_dist", 1.5)
        self.declare_parameter("angular_gain", 2.0)

        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.reward_desc_idx = self.get_parameter("reward_descriptor_index").value
        self.desc_dim = self.get_parameter("descriptor_dimension").value
        self.max_linear_vel = self.get_parameter("max_linear_vel").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.collect_approach_dist = self.get_parameter("collect_approach_dist").value
        self.frontier_min_size = self.get_parameter("frontier_min_size").value
        self.obstacle_slowdown_dist = self.get_parameter("obstacle_slowdown_dist").value
        self.lookahead_dist = self.get_parameter("lookahead_dist").value
        self.waypoint_reached_dist = self.get_parameter("waypoint_reached_dist").value
        self.replan_interval = self.get_parameter("replan_interval").value
        self.map_publish_interval = self.get_parameter("map_publish_interval").value
        self.safety_dist = self.get_parameter("safety_dist").value
        self.angular_gain = self.get_parameter("angular_gain").value

        # -- State --
        self.state = State.WAITING
        self.grid: QuadtreeOccupancyGrid | None = None
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_yaw = 0.0
        self.has_pose = False
        self.current_path: list[tuple[float, float]] = []
        self.path_idx = 0
        self.last_replan_time = 0.0
        self.last_map_publish_time = 0.0

        # Reward detection
        self.reward_visible = False
        self.reward_bearing = 0.0  # relative to agent heading
        self.reward_distance = float("inf")

        # Lidar data for obstacle checking
        self.latest_scan: LaserScan | None = None

        # -- QoS for latched topics --
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # -- Subscribers --
        self.create_subscription(
            Float32MultiArray, "/world_bounds", self._world_bounds_cb, latched_qos
        )
        self.create_subscription(Bool, "/episode_active", self._episode_active_cb, latched_qos)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/semantic_lidar", self._semantic_cb, 10
        )

        # -- Publishers --
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_map = self.create_publisher(OccupancyGrid, "/map", 10)
        self.pub_plan = self.create_publisher(Path, "/plan", 10)
        self.pub_frontiers = self.create_publisher(MarkerArray, "/frontiers", 10)
        self.pub_goal = self.create_publisher(Marker, "/goal_marker", 10)

        # -- Control loop timer (runs at ~10Hz but actual rate depends on sim) --
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info("ForagingExplorer node started, waiting for world bounds...")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _world_bounds_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            return
        w, h = msg.data[0], msg.data[1]
        self.get_logger().info(f"Received world bounds: {w} x {h}")
        self._init_grid(w, h)

    def _init_grid(self, width: float, height: float):
        self.world_width = width
        self.world_height = height
        self.grid = QuadtreeOccupancyGrid(
            world_width=width,
            world_height=height,
            min_resolution=self.grid_resolution,
        )
        self.current_path = []
        self.path_idx = 0
        self.state = State.EXPLORE
        self.get_logger().info(
            f"Initialized grid: {self.grid.cells_x}x{self.grid.cells_y} cells "
            f"at {self.grid_resolution}m resolution -> state=EXPLORE"
        )

    def _episode_active_cb(self, msg: Bool):
        self.get_logger().info(f"episode_active={msg.data}, state={self.state.name}, grid={'yes' if self.grid else 'no'}")
        if msg.data:
            if self.grid is not None:
                # New episode — clear and restart
                self.get_logger().info("New episode detected, resetting explorer state.")
                self.grid.clear()
                self.current_path = []
                self.path_idx = 0
                self.reward_visible = False
                self.has_pose = False
                self.state = State.EXPLORE
            else:
                # No grid yet — world_bounds not received.  Stay waiting.
                self.get_logger().warn("Episode active but no grid yet (world_bounds not received).")
        else:
            # Episode ended
            if self.state != State.WAITING:
                self.get_logger().info("Episode ended -> WAITING")
            self.state = State.WAITING
            self._publish_zero_vel()

    def _odom_cb(self, msg: Odometry):
        prev_has_pose = self.has_pose
        self.agent_x = msg.pose.pose.position.x
        self.agent_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw from quaternion (z-up convention)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.agent_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_pose = True
        if not prev_has_pose:
            self.get_logger().info(
                f"First pose received: x={self.agent_x:.1f}, y={self.agent_y:.1f}, "
                f"yaw={math.degrees(self.agent_yaw):.1f}deg"
            )

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        if self.grid is None or not self.has_pose:
            return

        n_rays = len(msg.ranges)
        valid_rays = sum(1 for r in msg.ranges if r > 0 and r < msg.range_max)

        # Log first scan
        if not hasattr(self, '_first_scan_logged'):
            self._first_scan_logged = True
            self.get_logger().info(
                f"First scan: {n_rays} rays, {valid_rays} valid, "
                f"angle_min={math.degrees(msg.angle_min):.1f}, "
                f"angle_inc={math.degrees(msg.angle_increment):.3f}, "
                f"range_max={msg.range_max:.1f}"
            )

        ranges = list(msg.ranges)
        self.grid.update_from_lidar(
            agent_x=self.agent_x,
            agent_y=self.agent_y,
            agent_yaw=self.agent_yaw,
            ranges=ranges,
            angle_start_rad=msg.angle_min,
            angle_increment_rad=msg.angle_increment,
            max_range=msg.range_max,
        )

        # Publish map (throttled)
        now = time.monotonic()
        if now - self.last_map_publish_time >= self.map_publish_interval:
            self.last_map_publish_time = now
            self._publish_map()

    def _semantic_cb(self, msg: Float32MultiArray):
        """Check semantic lidar for reward objects."""
        if not msg.data:
            self.reward_visible = False
            return

        if self.latest_scan is None:
            self.reward_visible = False
            return

        scan = self.latest_scan
        n_rays = len(scan.ranges)
        descriptors = msg.data

        if n_rays == 0:
            self.reward_visible = False
            return

        # Auto-detect descriptor dimension from scan ray count
        if len(descriptors) % n_rays != 0:
            if not hasattr(self, '_desc_mismatch_logged'):
                self._desc_mismatch_logged = True
                self.get_logger().warn(
                    f"Descriptor length {len(descriptors)} not divisible by "
                    f"n_rays {n_rays}, skipping semantic processing"
                )
            self.reward_visible = False
            return

        actual_desc_dim = len(descriptors) // n_rays
        if actual_desc_dim != self.desc_dim:
            if not hasattr(self, '_desc_dim_logged'):
                self._desc_dim_logged = True
                self.get_logger().info(
                    f"Auto-detected descriptor dimension: {actual_desc_dim} "
                    f"(was {self.desc_dim})"
                )
            self.desc_dim = actual_desc_dim

        if self.reward_desc_idx >= self.desc_dim:
            if not hasattr(self, '_desc_idx_warn_logged'):
                self._desc_idx_warn_logged = True
                self.get_logger().warn(
                    f"reward_descriptor_index={self.reward_desc_idx} >= "
                    f"descriptor_dimension={self.desc_dim}, can't detect rewards"
                )
            self.reward_visible = False
            return

        # The bridge publishes raw descriptors from Unity's Lidar2DMessage.
        # The LaserScan ranges are reversed (ROS CCW convention), but the
        # semantic descriptors arrive in Unity's original order (CW from
        # angleStartDeg).  To align them we reverse the descriptor array
        # per-ray so it matches the reversed ranges in LaserScan.
        desc_array = np.array(descriptors).reshape(n_rays, self.desc_dim)
        desc_array = desc_array[::-1]  # reverse to match LaserScan order

        ranges = np.array(scan.ranges)

        # Find rays where the reward descriptor index is active
        reward_mask = desc_array[:, self.reward_desc_idx] > 0.5

        # Also require the range to be a valid hit (not max range / no hit)
        valid_range = (ranges > 0) & (ranges < scan.range_max * 0.99)
        reward_hits = reward_mask & valid_range

        if not np.any(reward_hits):
            self.reward_visible = False
            return

        # Compute bearing to reward rays (in agent-local frame)
        reward_indices = np.where(reward_hits)[0]
        bearings = scan.angle_min + reward_indices * scan.angle_increment
        distances = ranges[reward_indices]

        # Weighted average bearing (closer = higher weight)
        weights = 1.0 / (distances + 0.1)
        self.reward_bearing = float(np.average(bearings, weights=weights))
        self.reward_distance = float(np.min(distances))
        self.reward_visible = True

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.state == State.WAITING or not self.has_pose or self.grid is None:
            # Log why we're not acting (throttled)
            if not hasattr(self, '_ctrl_wait_counter'):
                self._ctrl_wait_counter = 0
            self._ctrl_wait_counter += 1
            if self._ctrl_wait_counter % 50 == 1:
                # Check how many publishers exist on topics we subscribe to
                n_scan = self.count_publishers("/scan")
                n_odom = self.count_publishers("/odom")
                n_wb = self.count_publishers("/world_bounds")
                n_ea = self.count_publishers("/episode_active")
                self.get_logger().info(
                    f"Control loop waiting: state={self.state.name}, "
                    f"has_pose={self.has_pose}, grid={'yes' if self.grid else 'no'}, "
                    f"pubs: scan={n_scan}, odom={n_odom}, "
                    f"world_bounds={n_wb}, episode_active={n_ea}"
                )
            return

        # Log first active tick
        if not hasattr(self, '_first_ctrl_logged'):
            self._first_ctrl_logged = True
            self.get_logger().info(
                f"Control loop active! state={self.state.name}, "
                f"pos=({self.agent_x:.1f}, {self.agent_y:.1f})"
            )

        # -- Check for reward objects (transition to COLLECT) --
        if self.reward_visible and self.state == State.EXPLORE:
            self.state = State.COLLECT
            self.get_logger().info(
                f"Reward spotted! bearing={math.degrees(self.reward_bearing):.1f}deg "
                f"dist={self.reward_distance:.1f}m -> COLLECT"
            )

        if self.state == State.COLLECT:
            self._do_collect()
        elif self.state == State.EXPLORE:
            self._do_explore()

    def _do_collect(self):
        """Turn toward reward and approach it."""
        if not self.reward_visible:
            self.get_logger().info("Reward no longer visible -> EXPLORE")
            self.state = State.EXPLORE
            self.current_path = []  # force replan
            return

        # Proportional rotation toward reward bearing
        angle_error = self.reward_bearing  # already in agent-local frame
        angular_z = self._clamp(
            self.angular_gain * angle_error, -self.max_angular_vel, self.max_angular_vel
        )

        # Forward speed: reduce when not aligned, stop if obstacle too close
        alignment = max(0.0, math.cos(angle_error))
        linear_x = self.max_linear_vel * alignment

        # Obstacle check in forward cone
        min_front_range = self._get_min_front_range()
        if min_front_range < self.safety_dist:
            linear_x = 0.0
            # Still rotate toward reward
        elif min_front_range < self.obstacle_slowdown_dist:
            slow_factor = (min_front_range - self.safety_dist) / (
                self.obstacle_slowdown_dist - self.safety_dist
            )
            linear_x *= slow_factor

        self._publish_vel(linear_x, angular_z)

        # Publish goal marker at estimated reward position
        reward_wx = self.agent_x + self.reward_distance * math.cos(
            self.agent_yaw + self.reward_bearing
        )
        reward_wy = self.agent_y + self.reward_distance * math.sin(
            self.agent_yaw + self.reward_bearing
        )
        self._publish_goal_marker(reward_wx, reward_wy, r=1.0, g=0.8, b=0.0)

    def _do_explore(self):
        """Frontier-based exploration with A* path following."""
        now = time.monotonic()

        # Check if we need to replan
        need_replan = False
        if not self.current_path:
            need_replan = True
        elif self.path_idx >= len(self.current_path):
            need_replan = True
        elif now - self.last_replan_time > self.replan_interval:
            need_replan = True

        if need_replan:
            self.get_logger().info("Replanning...")
            self._plan_to_frontier()
            self.last_replan_time = now

        if not self.current_path:
            # No frontiers — nothing to explore
            self._publish_zero_vel()
            return

        # Advance path_idx past waypoints we've already reached
        while self.path_idx < len(self.current_path):
            wx, wy = self.current_path[self.path_idx]
            dist = math.hypot(wx - self.agent_x, wy - self.agent_y)
            if dist < self.waypoint_reached_dist:
                self.path_idx += 1
            else:
                break

        if self.path_idx >= len(self.current_path):
            # Reached end of path — will replan next tick
            self._publish_zero_vel()
            return

        # Pure pursuit: steer toward lookahead point on path
        target_x, target_y = self._get_lookahead_point()

        # Compute angle to target in world frame, then error relative to heading
        dx = target_x - self.agent_x
        dy = target_y - self.agent_y
        target_angle = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_angle - self.agent_yaw)

        angular_z = self._clamp(
            self.angular_gain * angle_error, -self.max_angular_vel, self.max_angular_vel
        )

        # Forward speed: proportional to alignment
        alignment = max(0.0, math.cos(angle_error))
        linear_x = self.max_linear_vel * alignment

        # Obstacle slowdown
        min_front_range = self._get_min_front_range()
        if min_front_range < self.safety_dist:
            linear_x = 0.0
            # Rotate in place to find a clear direction
            if abs(angle_error) < 0.3:
                angular_z = self.max_angular_vel  # turn away
        elif min_front_range < self.obstacle_slowdown_dist:
            slow_factor = (min_front_range - self.safety_dist) / (
                self.obstacle_slowdown_dist - self.safety_dist
            )
            linear_x *= slow_factor

        # Log velocity commands periodically
        if not hasattr(self, '_vel_log_counter'):
            self._vel_log_counter = 0
        self._vel_log_counter += 1
        if self._vel_log_counter % 20 == 1:
            self.get_logger().info(
                f"EXPLORE vel: lin={linear_x:.2f}, ang={angular_z:.2f}, "
                f"target=({target_x:.1f},{target_y:.1f}), "
                f"pos=({self.agent_x:.1f},{self.agent_y:.1f}), "
                f"yaw={math.degrees(self.agent_yaw):.1f}deg, "
                f"angle_err={math.degrees(angle_error):.1f}deg, "
                f"path_idx={self.path_idx}/{len(self.current_path)}, "
                f"front_range={min_front_range:.1f}"
            )

        self._publish_vel(linear_x, angular_z)

    def _plan_to_frontier(self):
        """Find frontiers, pick closest, plan A* path."""
        frontier_cells = self.grid.get_frontier_cells()
        if not frontier_cells:
            self.get_logger().info("No frontiers found.")
            self.current_path = []
            self._publish_frontiers([])
            return

        clusters = self.grid.cluster_frontiers(frontier_cells, self.frontier_min_size)
        if not clusters:
            self.get_logger().info("No frontier clusters large enough.")
            self.current_path = []
            self._publish_frontiers([])
            return

        # Pick closest cluster centroid
        best_cluster = None
        best_dist = float("inf")
        for cluster in clusters:
            cc, cr = QuadtreeOccupancyGrid.cluster_centroid(cluster)
            wx, wy = self.grid.cell_to_world(int(cc), int(cr))
            d = math.hypot(wx - self.agent_x, wy - self.agent_y)
            if d < best_dist:
                best_dist = d
                best_cluster = cluster

        if best_cluster is None:
            self.current_path = []
            return

        cc, cr = QuadtreeOccupancyGrid.cluster_centroid(best_cluster)
        goal_x, goal_y = self.grid.cell_to_world(int(cc), int(cr))

        # A* path
        path = self.grid.astar(
            self.agent_x, self.agent_y, goal_x, goal_y,
            inflation_radius=self.inflation_radius,
        )

        if path is None:
            self.get_logger().info(
                f"A* failed to frontier at ({goal_x:.0f}, {goal_y:.0f}), "
                f"trying next cluster..."
            )
            # Try other clusters
            for cluster in clusters:
                if cluster is best_cluster:
                    continue
                cc2, cr2 = QuadtreeOccupancyGrid.cluster_centroid(cluster)
                gx2, gy2 = self.grid.cell_to_world(int(cc2), int(cr2))
                path = self.grid.astar(
                    self.agent_x, self.agent_y, gx2, gy2,
                    inflation_radius=self.inflation_radius,
                )
                if path is not None:
                    goal_x, goal_y = gx2, gy2
                    break

        if path is None:
            self.get_logger().info("A* failed for all frontier clusters.")
            self.current_path = []
            return

        # Simplify path: skip waypoints that are very close together
        simplified = [path[0]]
        for wp in path[1:]:
            if math.hypot(wp[0] - simplified[-1][0], wp[1] - simplified[-1][1]) > self.grid_resolution * 2:
                simplified.append(wp)
        if simplified[-1] != path[-1]:
            simplified.append(path[-1])

        self.current_path = simplified
        self.path_idx = 0

        self.get_logger().info(
            f"Planned path to frontier ({goal_x:.0f}, {goal_y:.0f}), "
            f"{len(simplified)} waypoints, dist={best_dist:.0f}m"
        )

        # Publish visualizations
        self._publish_path(simplified)
        self._publish_frontiers(clusters)
        self._publish_goal_marker(goal_x, goal_y, r=0.0, g=1.0, b=0.0)

    # ------------------------------------------------------------------
    # Path following helpers
    # ------------------------------------------------------------------

    def _get_lookahead_point(self) -> tuple[float, float]:
        """Find the point on the path at lookahead_dist from the agent."""
        # Start from current path_idx
        for i in range(self.path_idx, len(self.current_path)):
            wx, wy = self.current_path[i]
            d = math.hypot(wx - self.agent_x, wy - self.agent_y)
            if d >= self.lookahead_dist:
                return wx, wy
        # If no point is far enough, return the last waypoint
        return self.current_path[-1]

    def _get_min_front_range(self) -> float:
        """Minimum range in the forward +-30 degree cone."""
        if self.latest_scan is None:
            return float("inf")
        scan = self.latest_scan
        min_range = float("inf")
        cone_half = math.radians(30)
        for i, r in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) <= cone_half and r > 0:
                min_range = min(min_range, r)
        return min_range

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def _publish_vel(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub_cmd_vel.publish(msg)

    def _publish_zero_vel(self):
        self._publish_vel(0.0, 0.0)

    def _publish_map(self):
        if self.grid is None:
            return
        msg = self.grid.to_occupancy_grid_msg(frame_id="odom")
        self.pub_map.publish(msg)

    def _publish_path(self, waypoints: list[tuple[float, float]]):
        msg = Path()
        msg.header = Header()
        msg.header.frame_id = "odom"
        for wx, wy in waypoints:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_plan.publish(msg)

    def _publish_frontiers(self, clusters: list[list[tuple[int, int]]]):
        ma = MarkerArray()

        # First, delete all previous markers
        delete_marker = Marker()
        delete_marker.header.frame_id = "odom"
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        for ci, cluster in enumerate(clusters):
            m = Marker()
            m.header.frame_id = "odom"
            m.ns = "frontiers"
            m.id = ci + 1  # offset by 1 since 0 is DELETEALL
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = self.grid_resolution
            m.scale.y = self.grid_resolution
            m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.6)
            m.pose.orientation.w = 1.0

            # Subsample if too many points
            step = max(1, len(cluster) // 200)
            for i in range(0, len(cluster), step):
                col, row = cluster[i]
                wx, wy = self.grid.cell_to_world(col, row)
                m.points.append(Point(x=wx, y=wy, z=0.1))

            ma.markers.append(m)

        self.pub_frontiers.publish(ma)

    def _publish_goal_marker(self, wx: float, wy: float, r=0.0, g=1.0, b=0.0):
        m = Marker()
        m.header.frame_id = "odom"
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.position.z = 1.0
        m.pose.orientation.w = 1.0
        m.scale.x = 3.0
        m.scale.y = 3.0
        m.scale.z = 3.0
        m.color = ColorRGBA(r=r, g=g, b=b, a=0.8)
        self.pub_goal.publish(m)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    @staticmethod
    def _normalize_angle(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = ForagingExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
