#!/usr/bin/env python3
"""Foraging exploration node.

Pure ROS2 node (no ratsim dependency) that builds an occupancy map from
lidar, detects frontiers, plans paths with A*, and follows them using
pure-pursuit (carrot-on-a-stick) path following.  When a reward object is
detected via semantic lidar descriptors, the agent switches to COLLECT
mode and approaches the reward.

Architecture, free mode (default): two wall timers at different rates:
  - Planning tick (4Hz): replans immediately when the current path is
    finished; otherwise at most every replan_interval seconds
  - Control timer (~50Hz): pure-pursuit path following / reward approach

Lockstep mode (lockstep:=true): no timers; each /step_end from the bridge
triggers plan (every replan_interval SIM seconds) + control, answered with
/cmd_vel_stamped, which the bridge blocks on before the next sim tick.

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
from collections import deque
from enum import Enum, auto

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from std_msgs.msg import Bool, Float32MultiArray, ColorRGBA, Header, Int32
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
        self.declare_parameter("obstacle_slowdown_dist", 1.5)
        self.declare_parameter("lookahead_dist", 5.0)     # max; shrinks with speed
        self.declare_parameter("min_lookahead", 0.8)
        self.declare_parameter("goal_reached_dist", 1.0)
        self.declare_parameter("path_clearance", 1.0)      # prefer cells this far from walls
        self.declare_parameter("path_clearance_weight", 3.0)
        self.declare_parameter("curve_speed_margin", 0.9)  # fraction of max omega usable via v*kappa
        self.declare_parameter("astar_max_expansions", 60000)  # cap per A* call (~1s worst case)
        self.declare_parameter("replan_interval", 2.0)     # seconds
        self.declare_parameter("map_publish_interval", 0.5)  # seconds
        self.declare_parameter("safety_dist", 1.5)
        self.declare_parameter("pure_rotation_threshold", 1.2)  # radians (~70 deg)
        # COLLECT survives this many consecutive frames without a reward hit
        # (sparse lidar misses small objects, especially at range)
        self.declare_parameter("reward_lost_ticks", 3)
        # Within this distance, drive straight at the reward instead of
        # following the planned path (we want to bump it, not stop short)
        self.declare_parameter("reward_approach_dist", 2.0)
        self.declare_parameter("control_rate", 50.0)       # Hz
        # Lockstep: drive planning/control from the bridge's /step_end instead
        # of wall timers, and answer every step with /cmd_vel_stamped so the
        # sim waits for the command. replan_interval then counts SIM seconds.
        self.declare_parameter("lockstep", True)

        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.reward_desc_idx = self.get_parameter("reward_descriptor_index").value
        self.desc_dim = self.get_parameter("descriptor_dimension").value
        self.max_linear_vel = self.get_parameter("max_linear_vel").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.frontier_min_size = self.get_parameter("frontier_min_size").value
        self.obstacle_slowdown_dist = self.get_parameter("obstacle_slowdown_dist").value
        self.lookahead_dist = self.get_parameter("lookahead_dist").value
        self.min_lookahead = self.get_parameter("min_lookahead").value
        self.goal_reached_dist = self.get_parameter("goal_reached_dist").value
        self.path_clearance = self.get_parameter("path_clearance").value
        self.path_clearance_weight = self.get_parameter("path_clearance_weight").value
        self.curve_speed_margin = self.get_parameter("curve_speed_margin").value
        self.astar_max_expansions = self.get_parameter("astar_max_expansions").value
        self.replan_interval = self.get_parameter("replan_interval").value
        self.map_publish_interval = self.get_parameter("map_publish_interval").value
        self.safety_dist = self.get_parameter("safety_dist").value
        self.pure_rotation_threshold = self.get_parameter("pure_rotation_threshold").value
        self.reward_lost_ticks = self.get_parameter("reward_lost_ticks").value
        self.reward_approach_dist = self.get_parameter("reward_approach_dist").value
        control_rate = self.get_parameter("control_rate").value
        self.lockstep = self.get_parameter("lockstep").value

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
        self._reward_world: tuple[float, float] | None = None  # last-seen position
        self._reward_missed_ticks = 0
        self._latest_descriptors: list | None = None  # this tick's semantic data
        self._reward_mask = None  # bool per ray (LaserScan order), this tick
        self._objects_collected = 0

        # Lidar data for obstacle checking
        self.latest_scan: LaserScan | None = None

        # Scan<->pose pairing: the bridge stamps the scan and odom of one sim
        # tick with the identical sim_time, so an exact stamp match pairs each
        # scan with the pose it was sensed at, regardless of arrival order.
        self._odom_by_stamp: dict[tuple[int, int], tuple[float, float, float]] = {}
        self._odom_stamps: deque[tuple[int, int]] = deque()
        self._pending_scans: deque[LaserScan] = deque(maxlen=20)

        # Lockstep bookkeeping
        self._last_cmd: tuple[float, float] = (0.0, 0.0)
        # Sim time (lockstep) or wall time (free mode) of the last planning run
        self._last_plan_time: float | None = None

        # Debug metrics (reset each episode)
        self._ep_start_pos: tuple[float, float] | None = None
        self._max_dist_from_start = 0.0
        self._n_plans = 0
        self._n_plan_failures = 0
        self._n_goals_reached = 0
        self._n_reflex_backups = 0
        self._last_dbg_time = 0.0
        self._fail_details: list[str] = []
        self._stuck_dumped = False

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
        self.create_subscription(Int32, "/objects_collected", self._objects_cb, 10)
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

        if self.lockstep:
            # Event-driven: one plan/control update per sim tick, answered
            # with a stamped command the bridge blocks on.
            self.pub_cmd_stamped = self.create_publisher(
                TwistStamped, "/cmd_vel_stamped", 10
            )
            self.create_subscription(Header, "/step_end", self._step_end_cb, 10)
        else:
            # -- Fast timer: pure-pursuit path following --
            self.create_timer(1.0 / control_rate, self._control_loop)

            # -- Planning: checked frequently so a finished path triggers an
            # immediate replan; a full replan_interval only gates mid-path replans.
            plan_period = max(0.05, min(0.25, self.replan_interval))
            self.create_timer(plan_period, self._planning_tick)

        self.get_logger().info(
            f"ForagingExplorer started: mode={'lockstep' if self.lockstep else 'free'}, "
            f"control={control_rate}Hz, "
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
        self._init_grid(w * 2, h * 2) # agent might not start at center of world, so make grid bigger

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
                # Stamps restart from 0 each episode — drop stale pairings
                self._odom_by_stamp.clear()
                self._odom_stamps.clear()
                self._pending_scans.clear()
                self._last_plan_time = None
                self._ep_start_pos = None
                self._max_dist_from_start = 0.0
                self._n_plans = 0
                self._n_plan_failures = 0
                self._n_goals_reached = 0
                self._n_reflex_backups = 0
                self._last_dbg_time = 0.0
                self._stuck_dumped = False
                self._reward_world = None
                self._reward_missed_ticks = 0
                self._latest_descriptors = None
                self._reward_mask = None
                self._objects_collected = 0
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
            self._odom_by_stamp.clear()
            self._odom_stamps.clear()
            self._pending_scans.clear()
            self._last_plan_time = None
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

        if self._ep_start_pos is None:
            self._ep_start_pos = (self.agent_x, self.agent_y)
        dfs = math.hypot(
            self.agent_x - self._ep_start_pos[0], self.agent_y - self._ep_start_pos[1]
        )
        if dfs > self._max_dist_from_start:
            self._max_dist_from_start = dfs

        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._odom_by_stamp[stamp] = (self.agent_x, self.agent_y, self.agent_yaw)
        self._odom_stamps.append(stamp)
        while len(self._odom_stamps) > 200:
            old = self._odom_stamps.popleft()
            self._odom_by_stamp.pop(old, None)

        # Integrate any scans that were waiting for this pose
        for _ in range(len(self._pending_scans)):
            scan, mask = self._pending_scans.popleft()
            s_stamp = (scan.header.stamp.sec, scan.header.stamp.nanosec)
            pose = self._odom_by_stamp.get(s_stamp)
            if pose is not None:
                self._integrate_scan(scan, pose, mask)
            else:
                self._pending_scans.append((scan, mask))

        if not prev_has_pose:
            self.get_logger().info(
                f"First pose: x={self.agent_x:.1f}, y={self.agent_y:.1f}, "
                f"yaw={math.degrees(self.agent_yaw):.1f}deg"
            )

    @staticmethod
    def _is_occlusion_scan(msg: LaserScan) -> bool:
        """True for the sensor's in-collider fault reading: every ray reports
        the same short range (occlusionDistance) with zero descriptors —
        e.g. the tick the agent bumps into a reward object.  That is fault
        simulation, not geometry; integrating it stamps a permanent ring of
        phantom obstacles around the bump site."""
        ranges = np.array(msg.ranges)
        if len(ranges) == 0 or np.any(ranges <= 0):
            return False
        return float(ranges.max() - ranges.min()) < 1e-4 and float(ranges[0]) < 1.0

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        self._detect_rewards(msg)
        if self.grid is None:
            return
        if self._is_occlusion_scan(msg):
            return  # don't map fault readings

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

        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        pose = self._odom_by_stamp.get(stamp)
        if pose is None:
            # Same-tick odom not delivered yet — integrate when it arrives
            self._pending_scans.append((msg, self._reward_mask))
            return
        self._integrate_scan(msg, pose, self._reward_mask)

    def _integrate_scan(
        self, msg: LaserScan, pose: tuple[float, float, float], reward_mask=None
    ):
        """Integrate a scan into the grid using the pose it was sensed at.

        Reward-class rays (reward_mask) never mark obstacles — rewards are
        collectibles the agent must be able to plan into and bump."""
        if self.grid is None:
            return
        px, py, pyaw = pose
        # The robot's own footprint is free evidence — clears phantom
        # obstacles stamped at bump contacts (collected rewards etc.)
        self.grid.clear_footprint(px, py)
        self.grid.update_from_lidar(
            agent_x=px,
            agent_y=py,
            agent_yaw=pyaw,
            ranges=list(msg.ranges),
            angle_start_rad=msg.angle_min,
            angle_increment_rad=msg.angle_increment,
            max_range=msg.range_max,
            skip_endpoint_mask=reward_mask,
        )

        # Publish map (throttled)
        now = time.monotonic()
        if now - self.last_map_publish_time >= self.map_publish_interval:
            self.last_map_publish_time = now
            self._publish_map()

    def _semantic_cb(self, msg: Float32MultiArray):
        """Store this tick's descriptors; detection runs at scan time so the
        mask pairs with the same tick's ranges (the bridge publishes semantic
        before scan)."""
        self._latest_descriptors = list(msg.data) if msg.data else None

    def _reward_not_seen(self):
        self.reward_visible = False
        self._reward_missed_ticks += 1

    def _detect_rewards(self, scan: LaserScan):
        """Detect reward objects in this tick's scan + stored descriptors.

        Sets reward_visible/bearing/distance, the per-ray reward mask (used
        to keep rewards out of the map and out of obstacle checks), and the
        last-seen reward world position."""
        self._reward_mask = None
        n_rays = len(scan.ranges)
        if n_rays == 0:
            self._reward_not_seen()
            return

        # Primary source: per-ray class ids in scan.intensities (0 = none,
        # class index + 1 otherwise) — same message as the ranges, so the
        # pairing is race-free. Fallback: /semantic_lidar descriptors
        # (older bridge; cross-topic ordering not guaranteed).
        class_mask = None
        if len(scan.intensities) == n_rays:
            intens = np.array(scan.intensities)
            class_mask = intens == float(self.reward_desc_idx + 1)
        else:
            descriptors = self._latest_descriptors
            self._latest_descriptors = None  # consume — never reuse
            if not descriptors:
                self._reward_not_seen()
                return
            if len(descriptors) % n_rays != 0:
                if not hasattr(self, '_desc_mismatch_logged'):
                    self._desc_mismatch_logged = True
                    self.get_logger().warn(
                        f"Descriptor length {len(descriptors)} not divisible "
                        f"by n_rays {n_rays}, skipping"
                    )
                self._reward_not_seen()
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
                self._reward_not_seen()
                return
            # Reverse descriptor array to match LaserScan ray order
            desc_array = np.array(descriptors).reshape(n_rays, self.desc_dim)
            desc_array = desc_array[::-1]
            class_mask = desc_array[:, self.reward_desc_idx] > 0.5

        ranges = np.array(scan.ranges)

        # Integration mask: every reward-class ray, no range cutoff — a
        # reward hit near max range must not map as an obstacle either.
        self._reward_mask = class_mask

        # Detection wants stable bearings: valid, non-extreme ranges only.
        valid_range = (ranges > 0) & (ranges < scan.range_max * 0.99)
        reward_hits = class_mask & valid_range

        if not np.any(reward_hits):
            self._reward_not_seen()
            return

        reward_indices = np.where(reward_hits)[0]
        bearings = scan.angle_min + reward_indices * scan.angle_increment
        distances = ranges[reward_indices]

        weights = 1.0 / (distances + 0.1)
        self.reward_bearing = float(np.average(bearings, weights=weights))
        self.reward_distance = float(np.min(distances))
        self.reward_visible = True
        self._reward_missed_ticks = 0

        # World position of the nearest reward hit.  agent_x/y/yaw are
        # same-tick: the bridge publishes odom before scan.
        j = int(np.argmin(distances))
        b = float(bearings[j])
        d = float(distances[j])
        self._reward_world = (
            self.agent_x + d * math.cos(self.agent_yaw + b),
            self.agent_y + d * math.sin(self.agent_yaw + b),
        )

    # ------------------------------------------------------------------
    # Lockstep step handler
    # ------------------------------------------------------------------

    def _step_end_cb(self, msg: Header):
        """One sim tick's sensor batch is complete: plan, act, reply.

        Odom and scan for this stamp were published before /step_end, so the
        scan is already integrated by the time this runs (same executor
        thread). The bridge blocks until it receives our /cmd_vel_stamped
        echoing this stamp, so this MUST reply on every path.
        """
        sim_time = msg.stamp.sec + msg.stamp.nanosec * 1e-9

        if self.state == State.WAITING or not self.has_pose or self.grid is None:
            self._publish_zero_vel()
        else:
            self._maybe_plan(sim_time)
            self._control_loop()

        # Periodic debug line (every 5 sim seconds)
        if sim_time - self._last_dbg_time >= 5.0:
            self._last_dbg_time = sim_time
            path_rem = max(0, len(self.current_path) - self.path_idx)
            self.get_logger().info(
                f"[dbg] t={sim_time:.1f}s pos=({self.agent_x:.1f},{self.agent_y:.1f}) "
                f"dfs={self._max_dist_from_start:.1f} state={self.state.name} "
                f"path_rem={path_rem} cmd=({self._last_cmd[0]:.2f},{self._last_cmd[1]:.2f}) "
                f"minfront={self._get_min_front_range():.1f} "
                f"plans={self._n_plans} fails={self._n_plan_failures} "
                f"goals={self._n_goals_reached} backups={self._n_reflex_backups} "
                f"obj={self._objects_collected}"
            )

        reply = TwistStamped()
        reply.header.stamp = msg.stamp
        reply.header.frame_id = msg.frame_id
        reply.twist.linear.x = self._last_cmd[0]
        reply.twist.angular.z = self._last_cmd[1]
        self.pub_cmd_stamped.publish(reply)

    # ------------------------------------------------------------------
    # Planning loop (slow timer)
    # ------------------------------------------------------------------

    def _planning_tick(self):
        """Free-mode planning timer body."""
        self._maybe_plan(time.monotonic())

    def _maybe_plan(self, now: float):
        """Plan when a replan was requested (path finished, reward lost, new
        episode) or when replan_interval elapsed. `now` is sim time in
        lockstep, wall time in free mode."""
        if (self._replan_requested
                or self._last_plan_time is None
                or now - self._last_plan_time >= self.replan_interval):
            self._last_plan_time = now
            self._planning_loop()

    def _planning_loop(self):
        """Detect frontiers, plan A* path."""
        if self.state not in (State.EXPLORE, State.COLLECT):
            return
        if not self.has_pose or self.grid is None:
            return

        # In COLLECT state, replan toward the reward, not frontiers — the
        # position refines with each new sighting.
        if self.state == State.COLLECT:
            if not self._plan_to_reward():
                self.get_logger().info("Reward unreachable -> EXPLORE")
                self._exit_collect()
            return

        self._plan_to_frontier()

    def _plan_to_frontier(self):
        """Find frontiers, plan A* to the closest reachable one."""
        # Clear here, not in _maybe_plan: an early-out in _planning_loop
        # (e.g. COLLECT) keeps the request pending.
        self._replan_requested = False
        self._n_plans += 1
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

        # Try clusters closest-first until one is reachable
        def centroid_world(cluster):
            cc, cr = QuadtreeOccupancyGrid.cluster_centroid(cluster)
            return self.grid.cell_to_world(int(cc), int(cr))

        candidates = sorted(
            (centroid_world(c) for c in clusters),
            key=lambda g: math.hypot(g[0] - self.agent_x, g[1] - self.agent_y),
        )

        # Goals already within reach are useless — "arriving" at one clears
        # nothing and would loop plan->reached->plan forever in place.
        min_goal_dist = self.goal_reached_dist + self.min_lookahead

        # Reachability filter: candidates outside the agent's connected
        # component of known-free space are unreachable (A* blocks unknown),
        # e.g. free-cell islands leaked behind walls.  Skipping them here
        # keeps them from burning the A*-failure budget every cycle.
        comp_labels = self.grid.get_free_components(self.inflation_radius)
        agent_comp = self._agent_component(comp_labels)

        path = None
        goal_x = goal_y = 0.0
        n_too_close = 0
        n_astar_fails = 0
        n_unretractable = 0
        n_islands = 0
        self._fail_details = []
        for raw_gx, raw_gy in candidates:
            # Retract the goal off the free/unknown boundary into known-free,
            # non-inflated space so the last stretch doesn't aim at a wall.
            retracted = self._retract_goal(raw_gx, raw_gy)
            if retracted is None:
                n_unretractable += 1
                path = None
                continue
            goal_x, goal_y = retracted
            g_col, g_row = self.grid.world_to_cell(goal_x, goal_y)
            if agent_comp == 0 or int(comp_labels[g_row, g_col]) != agent_comp:
                n_islands += 1
                path = None
                continue
            if math.hypot(goal_x - self.agent_x, goal_y - self.agent_y) < min_goal_dist:
                n_too_close += 1
                path = None
                continue
            path = self.grid.astar(
                self.agent_x, self.agent_y, goal_x, goal_y,
                inflation_radius=self.inflation_radius,
                clearance=self.path_clearance,
                clearance_weight=self.path_clearance_weight,
                max_expansions=self.astar_max_expansions,
            )
            if path is None:
                # Capped/unreachable goals are expensive — give up this cycle
                # after a few and retry on the replan interval.
                n_astar_fails += 1
                st = self.grid.last_astar_stats
                self._fail_details.append(
                    f"goal=({goal_x:.0f},{goal_y:.0f}) "
                    f"d={math.hypot(goal_x - self.agent_x, goal_y - self.agent_y):.0f}m "
                    f"{st.get('reason')} exp={st.get('expansions')} "
                    f"sval={st.get('start_inflated_val')}"
                )
                if n_astar_fails >= 4:
                    break
            if path is not None:
                if (goal_x, goal_y) != (raw_gx, raw_gy):
                    self.get_logger().info(
                        f"Goal retracted ({raw_gx:.1f},{raw_gy:.1f}) -> "
                        f"({goal_x:.1f},{goal_y:.1f})"
                    )
                break

        if path is None:
            self._n_plan_failures += 1
            self.get_logger().info(
                f"No plan: {len(candidates)} clusters "
                f"({n_too_close} within {min_goal_dist:.1f}m, "
                f"{n_unretractable} unretractable, {n_islands} islands, "
                f"rest unreachable). " + "; ".join(self._fail_details)
            )
            self._dump_stuck_state(candidates)
            self.current_path = []
            return

        self.current_path = path
        self.path_idx = 0

        dist = math.hypot(goal_x - self.agent_x, goal_y - self.agent_y)
        self.get_logger().info(
            f"Planned path to frontier ({goal_x:.0f}, {goal_y:.0f}), "
            f"{len(path)} waypoints, dist={dist:.0f}m, "
            f"{len(clusters)} clusters"
        )

        self._publish_path(path)
        self._publish_frontiers(clusters)
        self._publish_goal_marker(goal_x, goal_y, r=0.0, g=1.0, b=0.0)

    def _agent_component(self, comp_labels) -> int:
        """The agent's connected component of known-free space (0 = none).

        If the agent's own cell is not free (inflated band / unscanned),
        use the nearest free cell's component. The search radius must
        comfortably exceed the inflation radius: a phantom obstacle near
        the agent pushes the nearest free cell out by its whole inflation
        ring, and returning 0 here makes every goal look unreachable."""
        a_col, a_row = self.grid.world_to_cell(self.agent_x, self.agent_y)
        if self.grid._in_bounds(a_col, a_row):
            comp = int(comp_labels[a_row, a_col])
            if comp != 0:
                return comp
        esc = max(
            int(math.ceil(self.inflation_radius / self.grid_resolution)) + 1,
            int(math.ceil(1.5 / self.grid_resolution)),
        )
        best_d2 = None
        agent_comp = 0
        for dr in range(-esc, esc + 1):
            for dc in range(-esc, esc + 1):
                c, r = a_col + dc, a_row + dr
                d2 = dr * dr + dc * dc
                if (d2 <= esc * esc and self.grid._in_bounds(c, r)
                        and comp_labels[r, c] != 0
                        and (best_d2 is None or d2 < best_d2)):
                    best_d2 = d2
                    agent_comp = int(comp_labels[r, c])
        return agent_comp

    def _dump_stuck_state(self, candidates) -> None:
        """One-shot per episode: save the grid + candidates when a planning
        cycle fails completely, for offline analysis."""
        if self._stuck_dumped:
            return
        self._stuck_dumped = True
        try:
            import os
            out_dir = os.environ.get("RATSIM_DEBUG_DIR", "/tmp")
            path = os.path.join(out_dir, f"stuck_dump_{int(time.time())}.npz")
            np.savez_compressed(
                path,
                flat=self.grid.to_flat_grid(),
                inflated=self.grid.get_inflated_grid(self.inflation_radius),
                agent=np.array([self.agent_x, self.agent_y]),
                candidates=np.array(candidates, dtype=np.float64),
                origin=np.array([self.grid.origin_x, self.grid.origin_y]),
                resolution=np.array([self.grid.min_resolution]),
            )
            self.get_logger().info(f"Stuck-state dump written: {path}")
        except Exception as e:  # debug aid only — never kill planning
            self.get_logger().warn(f"Stuck-state dump failed: {e}")

    def _retract_goal(self, gx: float, gy: float) -> tuple[float, float] | None:
        """Move a frontier goal into known-free, non-inflated space.

        Steps from the goal toward the agent; if that fails, scans a small
        box around the goal. Returns None if no known-free, non-inflated
        cell is found — A* blocks unknown cells, so planning to such a goal
        would be a guaranteed failure."""
        inflated = self.grid.get_inflated_grid(self.inflation_radius)

        def ok(wx, wy):
            c, r = self.grid.world_to_cell(wx, wy)
            return self.grid._in_bounds(c, r) and inflated[r, c] == 0  # FREE

        if ok(gx, gy):
            return gx, gy

        # Walk toward the agent
        dx = self.agent_x - gx
        dy = self.agent_y - gy
        d = math.hypot(dx, dy)
        if d > 1e-6:
            step = self.grid_resolution / 2.0
            max_retract = self.inflation_radius + 1.0
            n_steps = int(max_retract / step)
            for i in range(1, n_steps + 1):
                wx = gx + dx / d * step * i
                wy = gy + dy / d * step * i
                if ok(wx, wy):
                    return wx, wy

        # Box scan around the goal, nearest cell first
        r_cells = int(math.ceil(1.5 / self.grid_resolution))
        gc, gr = self.grid.world_to_cell(gx, gy)
        best = None
        best_d2 = float("inf")
        for drow in range(-r_cells, r_cells + 1):
            for dcol in range(-r_cells, r_cells + 1):
                c, r = gc + dcol, gr + drow
                if not self.grid._in_bounds(c, r) or inflated[r, c] != 0:
                    continue
                d2 = drow * drow + dcol * dcol
                if d2 < best_d2:
                    best_d2 = d2
                    best = (c, r)
        if best is not None:
            return self.grid.cell_to_world(best[0], best[1])
        return None

    # ------------------------------------------------------------------
    # Control loop (fast timer) — pure pursuit path following
    # ------------------------------------------------------------------

    def _control_loop(self):
        if self.state == State.WAITING or not self.has_pose or self.grid is None:
            return

        # Check for reward objects → transition to COLLECT (only if the
        # reward is reachable through known-free space; otherwise keep
        # exploring until the connecting corridor is mapped)
        if self.reward_visible and self.state == State.EXPLORE:
            if self._plan_to_reward():
                self.state = State.COLLECT
                self.get_logger().info(
                    f"Reward spotted! bearing={math.degrees(self.reward_bearing):.1f}deg "
                    f"dist={self.reward_distance:.1f}m -> COLLECT"
                )

        if self.state == State.COLLECT:
            self._do_collect()
        elif self.state == State.EXPLORE:
            self._do_explore()

    def _exit_collect(self):
        """Back to EXPLORE; forget the reward and ask for a frontier plan."""
        self.state = State.EXPLORE
        self._reward_world = None
        self._reward_missed_ticks = 0
        self.current_path = []
        self.path_idx = 0
        self._replan_requested = True

    def _objects_cb(self, msg: Int32):
        prev = self._objects_collected
        self._objects_collected = msg.data
        if msg.data > prev and self.state == State.COLLECT:
            self.get_logger().info(
                f"Reward collected! total={msg.data} -> EXPLORE"
            )
            self._exit_collect()

    def _do_collect(self):
        """Navigate into the last-seen reward: planned path while far,
        direct chase inside reward_approach_dist (we want to bump it).
        Gives up after reward_lost_ticks consecutive frames without a
        sighting — the sparse lidar misses small objects routinely."""
        if (self._reward_world is None
                or self._reward_missed_ticks > self.reward_lost_ticks):
            self.get_logger().info(
                f"Reward lost ({self._reward_missed_ticks} ticks) -> EXPLORE"
            )
            self._exit_collect()
            self._publish_zero_vel()
            return

        rx, ry = self._reward_world
        dx = rx - self.agent_x
        dy = ry - self.agent_y
        dist = math.hypot(dx, dy)

        if dist > self.reward_approach_dist:
            # Far: follow the planned path (replanned on the usual interval
            # as sightings refine the reward position)
            if not self.current_path:
                if not self._plan_to_reward():
                    self.get_logger().info("Reward unreachable -> EXPLORE")
                    self._exit_collect()
                    self._publish_zero_vel()
                    return
            self._follow_path_step()
            return

        # Close: drive straight at the last-known point — no goal radius,
        # the collision with the object IS the goal.
        angle_error = math.atan2(dy, dx) - self.agent_yaw
        angle_error = (angle_error + math.pi) % (2.0 * math.pi) - math.pi

        if abs(angle_error) > self.pure_rotation_threshold:
            omega = self.max_angular_vel if angle_error > 0 else -self.max_angular_vel
            self._publish_vel(0.0, omega)
        else:
            d_l = max(dist, 0.1)
            delta_y = d_l * math.sin(angle_error)
            alignment = max(0.0, math.cos(angle_error))
            v = self.max_linear_vel * (0.3 + 0.7 * alignment)
            omega = 2.0 * v * delta_y / (d_l * d_l)
            omega = self._clamp(omega, -self.max_angular_vel, self.max_angular_vel)
            # Obstacle check still applies — reward rays are excluded from it
            linear_x, omega = self._apply_obstacle_avoidance(v, omega)
            self._publish_vel(linear_x, omega)

        self._publish_goal_marker(rx, ry, r=1.0, g=0.8, b=0.0)

    def _plan_to_reward(self) -> bool:
        """Plan a path to the last-seen reward position. False when the
        reward is not (yet) reachable through known-free space."""
        if self._reward_world is None or self.grid is None or not self.has_pose:
            return False
        self._replan_requested = False
        rx, ry = self._reward_world
        retracted = self._retract_goal(rx, ry)
        if retracted is None:
            return False
        gx, gy = retracted
        comp_labels = self.grid.get_free_components(self.inflation_radius)
        agent_comp = self._agent_component(comp_labels)
        g_col, g_row = self.grid.world_to_cell(gx, gy)
        if agent_comp == 0 or int(comp_labels[g_row, g_col]) != agent_comp:
            return False
        path = self.grid.astar(
            self.agent_x, self.agent_y, gx, gy,
            inflation_radius=self.inflation_radius,
            clearance=self.path_clearance,
            clearance_weight=self.path_clearance_weight,
            max_expansions=self.astar_max_expansions,
        )
        if path is None:
            return False
        self.current_path = path
        self.path_idx = 0
        self._publish_path(path)
        self._publish_goal_marker(gx, gy, r=1.0, g=0.8, b=0.0)
        return True

    def _do_explore(self):
        """Frontier path following: stop within goal_reached_dist of the
        path end (the retracted goal sits in free space, no need to defend
        the last meter), otherwise pure-pursuit along the path."""
        if not self.current_path:
            self._publish_zero_vel()
            return

        end_x, end_y = self.current_path[-1]
        if math.hypot(end_x - self.agent_x, end_y - self.agent_y) < self.goal_reached_dist:
            self._n_goals_reached += 1
            self.current_path = []
            self._replan_requested = True
            self._publish_zero_vel()
            return

        self._follow_path_step()

    def _follow_path_step(self):
        """Pure-pursuit (carrot-on-a-stick) along current_path. Shared by
        EXPLORE (with a goal-reached stop) and COLLECT (without one)."""
        if not self.current_path:
            self._publish_zero_vel()
            return

        # Advance path_idx past reached waypoints (monotonic — never goes backward)
        while self.path_idx < len(self.current_path):
            wx, wy = self.current_path[self.path_idx]
            dist = math.hypot(wx - self.agent_x, wy - self.agent_y)
            if dist < self.grid_resolution:
                self.path_idx += 1
            else:
                break

        if self.path_idx >= len(self.current_path):
            # Path finished — ask for a fresh plan instead of parking until
            # the next scheduled replan. (A failed plan empties current_path
            # without setting this, so failures still retry on the interval.)
            self._replan_requested = True
            self._publish_zero_vel()
            return

        # Find the carrot: lookahead point on the path at distance d_l
        carrot_x, carrot_y, anchor_idx = self._find_carrot()

        # Anchor (closest path point) never goes backward along the path
        if anchor_idx > self.path_idx:
            self.path_idx = anchor_idx

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

        angle_to_carrot = math.atan2(local_y, local_x)

        # Pure rotation: if carrot is behind us, rotate in place first
        if abs(angle_to_carrot) > self.pure_rotation_threshold:
            omega = self.max_angular_vel if angle_to_carrot > 0 else -self.max_angular_vel
            self._publish_vel(0.0, omega)
            self._publish_carrot_marker(carrot_x, carrot_y)
            return

        # Pure pursuit: curvature κ = 2·Δy / d_l²,  ω = v · κ
        delta_y = local_y
        curvature = 2.0 * delta_y / (d_l * d_l)

        # Curvature-limited speed: only as fast as the omega cap can steer.
        # As demanded curvature grows, v -> 0 and this blends into a pivot.
        v = self.max_linear_vel
        if abs(curvature) > 1e-6:
            v = min(v, self.curve_speed_margin * self.max_angular_vel / abs(curvature))

        omega = v * curvature
        omega = self._clamp(omega, -self.max_angular_vel, self.max_angular_vel)

        # Obstacle check
        linear_x, omega = self._apply_obstacle_avoidance(v, omega)

        self._publish_vel(linear_x, omega)

        # Publish carrot marker
        self._publish_carrot_marker(carrot_x, carrot_y)

    def _find_carrot(self) -> tuple[float, float, int]:
        """Find the carrot: the point at lookahead arc length ALONG the path
        ahead of the closest path point (walls-aware — never jumps across a
        corner the way straight-line lookahead does).

        Lookahead scales with the last commanded speed so tight sections are
        tracked tightly. Returns (carrot_x, carrot_y, anchor_index) where
        anchor_index is the closest path point (monotonic path progress).
        """
        path = self.current_path
        ax, ay = self.agent_x, self.agent_y
        d_l = self._clamp(
            0.7 * max(self._last_cmd[0], 0.0), self.min_lookahead, self.lookahead_dist
        )

        # Re-anchor: closest path point, searching forward from path_idx
        end = min(len(path), self.path_idx + 200)
        anchor = self.path_idx
        best_d = float("inf")
        for i in range(self.path_idx, end):
            d = math.hypot(path[i][0] - ax, path[i][1] - ay)
            if d < best_d:
                best_d = d
                anchor = i

        # Walk arc length along the path from the anchor
        s = 0.0
        px, py = path[anchor]
        for i in range(anchor + 1, len(path)):
            wx, wy = path[i]
            seg = math.hypot(wx - px, wy - py)
            if s + seg >= d_l and seg > 1e-9:
                t = (d_l - s) / seg
                return (px + t * (wx - px), py + t * (wy - py), anchor)
            s += seg
            px, py = wx, wy

        # Path shorter than lookahead — carrot is the last point
        return (path[-1][0], path[-1][1], anchor)

    def _apply_obstacle_avoidance(
        self, v: float, omega: float
    ) -> tuple[float, float]:
        """Reduce speed or back up based on front obstacle distance."""
        min_front = self._get_min_front_range()
        if min_front < self.safety_dist:
            # Back up and turn away from closest obstacle side
            self._n_reflex_backups += 1
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
        mask = self._reward_mask  # same-tick, LaserScan ray order
        # Sum up inverse-range contributions from left vs right
        left_weight = 0.0
        right_weight = 0.0
        for i, r in enumerate(scan.ranges):
            if r <= 0:
                continue
            if mask is not None and i < len(mask) and mask[i]:
                continue  # rewards are collectibles, not obstacles
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
        mask = self._reward_mask  # same-tick, LaserScan ray order
        min_range = float("inf")
        cone_half = math.radians(30)
        for i, r in enumerate(scan.ranges):
            if mask is not None and i < len(mask) and mask[i]:
                continue  # rewards are collectibles, not obstacles
            angle = scan.angle_min + i * scan.angle_increment
            if abs(angle) <= cone_half and r > 0:
                min_range = min(min_range, r)
        return min_range

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def _publish_vel(self, linear_x: float, angular_z: float):
        self._last_cmd = (linear_x, angular_z)
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
