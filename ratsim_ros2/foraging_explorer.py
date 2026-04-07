#!/usr/bin/env python3
"""Foraging exploration node.

Pure ROS2 node (no ratsim dependency) that builds an occupancy map from
lidar, detects frontiers, plans paths with A*, and follows them using
pure-pursuit (carrot-on-a-stick) path following.  When a reward object is
detected via semantic lidar descriptors, the agent switches to COLLECT
mode and approaches the reward.

Architecture: two timers run at different rates:
  - Planning timer (~0.5Hz): frontier detection, clustering, A* path planning
  - Control timer (~50Hz): pure-pursuit path following / reward approach

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
        self.declare_parameter("max_angular_vel", 2.0)
        self.declare_parameter("frontier_min_size", 5)
        self.declare_parameter("obstacle_slowdown_dist", 5.0)
        self.declare_parameter("lookahead_dist", 5.0)
        self.declare_parameter("replan_interval", 2.0)     # seconds
        self.declare_parameter("map_publish_interval", 0.5)  # seconds
        self.declare_parameter("safety_dist", 1.5)
        self.declare_parameter("control_rate", 50.0)       # Hz

        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.reward_desc_idx = self.get_parameter("reward_descriptor_index").value
        self.desc_dim = self.get_parameter("descriptor_dimension").value
        self.max_linear_vel = self.get_parameter("max_linear_vel").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.frontier_min_size = self.get_parameter("frontier_min_size").value
        self.obstacle_slowdown_dist = self.get_parameter("obstacle_slowdown_dist").value
        self.lookahead_dist = self.get_parameter("lookahead_dist").value
        self.replan_interval = self.get_parameter("replan_interval").value
        self.map_publish_interval = self.get_parameter("map_publish_interval").value
        self.safety_dist = self.get_parameter("safety_dist").value
        control_rate = self.get_parameter("control_rate").value

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
        self._replan_requested = False

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
        self.pub_carrot = self.create_publisher(Marker, "/carrot_marker", 10)

        # -- Fast timer: pure-pursuit path following --
        self.create_timer(1.0 / control_rate, self._control_loop)

        # -- Slow timer: frontier detection + A* planning --
        self.create_timer(self.replan_interval, self._planning_loop)

        self.get_logger().info(
            f"ForagingExplorer started: control={control_rate}Hz, "
            f"replan_interval={self.replan_interval}s, "
            f"lookahead={self.lookahead_dist}m"
        )

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
        self._replan_requested = True
        self.get_logger().info(
            f"Initialized grid: {self.grid.cells_x}x{self.grid.cells_y} cells "
            f"at {self.grid_resolution}m resolution -> state=EXPLORE"
        )

    def _episode_active_cb(self, msg: Bool):
        self.get_logger().info(
            f"episode_active={msg.data}, state={self.state.name}, "
            f"grid={'yes' if self.grid else 'no'}"
        )
        if msg.data:
            if self.grid is not None:
                self.get_logger().info("New episode detected, resetting explorer state.")
                self.grid.clear()
                self.current_path = []
                self.path_idx = 0
                self.reward_visible = False
                self.has_pose = False
                self.state = State.EXPLORE
                self._replan_requested = True
            else:
                self.get_logger().warn(
                    "Episode active but no grid yet (world_bounds not received)."
                )
        else:
            if self.state != State.WAITING:
                self.get_logger().info("Episode ended -> WAITING")
            self.state = State.WAITING
            self._publish_zero_vel()

    def _odom_cb(self, msg: Odometry):
        prev_has_pose = self.has_pose
        self.agent_x = msg.pose.pose.position.x
        self.agent_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.agent_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_pose = True
        if not prev_has_pose:
            self.get_logger().info(
                f"First pose: x={self.agent_x:.1f}, y={self.agent_y:.1f}, "
                f"yaw={math.degrees(self.agent_yaw):.1f}deg"
            )

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        if self.grid is None or not self.has_pose:
            return

        # Log first scan
        if not hasattr(self, '_first_scan_logged'):
            self._first_scan_logged = True
            n_rays = len(msg.ranges)
            valid_rays = sum(1 for r in msg.ranges if r > 0 and r < msg.range_max)
            self.get_logger().info(
                f"First scan: {n_rays} rays, {valid_rays} valid, "
                f"angle_min={math.degrees(msg.angle_min):.1f}, "
                f"angle_inc={math.degrees(msg.angle_increment):.3f}, "
                f"range_max={msg.range_max:.1f}"
            )

        self.grid.update_from_lidar(
            agent_x=self.agent_x,
            agent_y=self.agent_y,
            agent_yaw=self.agent_yaw,
            ranges=list(msg.ranges),
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
        if not msg.data or self.latest_scan is None:
            self.reward_visible = False
            return

        scan = self.latest_scan
        n_rays = len(scan.ranges)
        descriptors = msg.data

        if n_rays == 0:
            self.reward_visible = False
            return

        # Auto-detect descriptor dimension
        if len(descriptors) % n_rays != 0:
            if not hasattr(self, '_desc_mismatch_logged'):
                self._desc_mismatch_logged = True
                self.get_logger().warn(
                    f"Descriptor length {len(descriptors)} not divisible by "
                    f"n_rays {n_rays}, skipping"
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

        # Reverse descriptor array to match LaserScan ray order
        desc_array = np.array(descriptors).reshape(n_rays, self.desc_dim)
        desc_array = desc_array[::-1]

        ranges = np.array(scan.ranges)

        reward_mask = desc_array[:, self.reward_desc_idx] > 0.5
        valid_range = (ranges > 0) & (ranges < scan.range_max * 0.99)
        reward_hits = reward_mask & valid_range

        if not np.any(reward_hits):
            self.reward_visible = False
            return

        reward_indices = np.where(reward_hits)[0]
        bearings = scan.angle_min + reward_indices * scan.angle_increment
        distances = ranges[reward_indices]

        weights = 1.0 / (distances + 0.1)
        self.reward_bearing = float(np.average(bearings, weights=weights))
        self.reward_distance = float(np.min(distances))
        self.reward_visible = True

    # ------------------------------------------------------------------
    # Planning loop (slow timer)
    # ------------------------------------------------------------------

    def _planning_loop(self):
        """Runs at low frequency: detect frontiers, plan A* path."""
        if self.state not in (State.EXPLORE, State.COLLECT):
            return
        if not self.has_pose or self.grid is None:
            return

        # In COLLECT state, don't replan frontiers — just follow reward
        if self.state == State.COLLECT:
            return

        self._plan_to_frontier()

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

        self.current_path = path
        self.path_idx = 0

        self.get_logger().info(
            f"Planned path to frontier ({goal_x:.0f}, {goal_y:.0f}), "
            f"{len(path)} waypoints, dist={best_dist:.0f}m"
        )

        self._publish_path(path)
        self._publish_frontiers(clusters)
        self._publish_goal_marker(goal_x, goal_y, r=0.0, g=1.0, b=0.0)

    # ------------------------------------------------------------------
    # Control loop (fast timer) — pure pursuit path following
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.state == State.WAITING or not self.has_pose or self.grid is None:
            return

        # Check for reward objects → transition to COLLECT
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
        """Pure-pursuit toward reward bearing."""
        if not self.reward_visible:
            self.get_logger().info("Reward no longer visible -> EXPLORE")
            self.state = State.EXPLORE
            self.current_path = []
            self._replan_requested = True
            return

        # Treat the reward as a virtual lookahead point in local frame
        angle_error = self.reward_bearing
        d_l = max(self.reward_distance, 0.1)

        # Lateral offset in robot-local frame
        delta_y = d_l * math.sin(angle_error)

        # Pure pursuit: ω = 2·v·Δy / d_l²
        alignment = max(0.0, math.cos(angle_error))
        v = self.max_linear_vel * (0.3 + 0.7 * alignment)
        omega = 2.0 * v * delta_y / (d_l * d_l)
        omega = self._clamp(omega, -self.max_angular_vel, self.max_angular_vel)

        # Obstacle check
        linear_x, omega = self._apply_obstacle_avoidance(v, omega)

        self._publish_vel(linear_x, omega)

        # Publish goal marker
        reward_wx = self.agent_x + self.reward_distance * math.cos(
            self.agent_yaw + self.reward_bearing
        )
        reward_wy = self.agent_y + self.reward_distance * math.sin(
            self.agent_yaw + self.reward_bearing
        )
        self._publish_goal_marker(reward_wx, reward_wy, r=1.0, g=0.8, b=0.0)

    def _do_explore(self):
        """Pure-pursuit (carrot-on-a-stick) path following."""
        if not self.current_path:
            self._publish_zero_vel()
            return

        # Advance path_idx past reached waypoints
        while self.path_idx < len(self.current_path):
            wx, wy = self.current_path[self.path_idx]
            dist = math.hypot(wx - self.agent_x, wy - self.agent_y)
            if dist < self.grid_resolution:
                self.path_idx += 1
            else:
                break

        if self.path_idx >= len(self.current_path):
            self._publish_zero_vel()
            return

        # Find the carrot: lookahead point on the path at distance d_l
        carrot_x, carrot_y = self._find_carrot()

        # Transform carrot to robot-local frame
        dx = carrot_x - self.agent_x
        dy = carrot_y - self.agent_y
        # Rotate into robot frame (x=forward, y=left)
        local_x = math.cos(self.agent_yaw) * dx + math.sin(self.agent_yaw) * dy
        local_y = -math.sin(self.agent_yaw) * dx + math.cos(self.agent_yaw) * dy

        d_l = math.hypot(local_x, local_y)
        if d_l < 0.01:
            self._publish_zero_vel()
            return

        # Pure pursuit: curvature κ = 2·Δy / d_l²,  ω = v · κ
        # Δy is the lateral offset in robot frame
        delta_y = local_y
        curvature = 2.0 * delta_y / (d_l * d_l)

        # Reduce forward speed when high curvature is needed (tight turns)
        # angle_to_carrot in local frame: atan2(local_y, local_x)
        angle_to_carrot = math.atan2(local_y, local_x)
        alignment = max(0.0, math.cos(angle_to_carrot))
        v = self.max_linear_vel * (0.3 + 0.7 * alignment)

        omega = v * curvature
        omega = self._clamp(omega, -self.max_angular_vel, self.max_angular_vel)

        # Obstacle check
        linear_x, omega = self._apply_obstacle_avoidance(v, omega)

        self._publish_vel(linear_x, omega)

        # Publish carrot marker
        self._publish_carrot_marker(carrot_x, carrot_y)

    def _find_carrot(self) -> tuple[float, float]:
        """Find the carrot point: first path point at least lookahead_dist away.

        Interpolates between waypoints for a smooth carrot position.
        """
        path = self.current_path
        ax, ay = self.agent_x, self.agent_y
        d_l = self.lookahead_dist

        # Walk along the path from path_idx, accumulating distance
        for i in range(self.path_idx, len(path)):
            wx, wy = path[i]
            dist = math.hypot(wx - ax, wy - ay)
            if dist >= d_l:
                # Interpolate between previous point and this one
                if i > self.path_idx:
                    px, py = path[i - 1]
                    prev_dist = math.hypot(px - ax, py - ay)
                    seg_len = math.hypot(wx - px, wy - py)
                    if seg_len > 0.01:
                        # How far along this segment to reach d_l
                        remaining = d_l - prev_dist
                        t = self._clamp(remaining / seg_len, 0.0, 1.0)
                        return (px + t * (wx - px), py + t * (wy - py))
                return wx, wy

        # Path is shorter than lookahead — return last point
        return path[-1]

    def _apply_obstacle_avoidance(
        self, v: float, omega: float
    ) -> tuple[float, float]:
        """Reduce speed or back up based on front obstacle distance."""
        min_front = self._get_min_front_range()
        if min_front < self.safety_dist:
            # Back up and turn away from closest obstacle side
            v = -1.0
            # Turn away from the side with the closest obstacle
            omega = self._get_avoidance_omega()
        elif min_front < self.obstacle_slowdown_dist:
            factor = (min_front - self.safety_dist) / (
                self.obstacle_slowdown_dist - self.safety_dist
            )
            v *= factor
        return v, omega

    def _get_avoidance_omega(self) -> float:
        """Compute angular velocity to turn away from the closest obstacle."""
        if self.latest_scan is None:
            return self.max_angular_vel
        scan = self.latest_scan
        # Sum up inverse-range contributions from left vs right
        left_weight = 0.0
        right_weight = 0.0
        for i, r in enumerate(scan.ranges):
            if r <= 0:
                continue
            angle = scan.angle_min + i * scan.angle_increment
            w = 1.0 / (r + 0.1)
            if angle > 0:
                left_weight += w
            else:
                right_weight += w
        # Turn away from the heavier side
        if left_weight > right_weight:
            return -self.max_angular_vel  # turn right
        return self.max_angular_vel  # turn left

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

        delete_marker = Marker()
        delete_marker.header.frame_id = "odom"
        delete_marker.action = Marker.DELETEALL
        ma.markers.append(delete_marker)

        for ci, cluster in enumerate(clusters):
            m = Marker()
            m.header.frame_id = "odom"
            m.ns = "frontiers"
            m.id = ci + 1
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = self.grid_resolution
            m.scale.y = self.grid_resolution
            m.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.6)
            m.pose.orientation.w = 1.0

            step = max(1, len(cluster) // 200)
            for i in range(0, len(cluster), step):
                col, row = cluster[i]
                wx, wy = self.grid.cell_to_world(col, row)
                m.points.append(Point(x=wx, y=wy, z=0.1))

            ma.markers.append(m)

        self.pub_frontiers.publish(ma)

    def _publish_carrot_marker(self, wx: float, wy: float):
        m = Marker()
        m.header.frame_id = "odom"
        m.ns = "carrot"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = wx
        m.pose.position.y = wy
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = 1.5
        m.scale.y = 1.5
        m.scale.z = 1.5
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9)  # orange
        self.pub_carrot.publish(m)

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
