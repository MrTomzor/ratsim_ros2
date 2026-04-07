"""Quadtree-based 2D occupancy grid for frontier exploration.

Cell values follow nav_msgs/OccupancyGrid convention:
  -1  = UNKNOWN
   0  = FREE
  100 = OCCUPIED
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Optional

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


# ---------------------------------------------------------------------------
# Bresenham line rasterization (from ratsim/nav/occupancy_mapping_planning.py)
# ---------------------------------------------------------------------------

def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Integer coordinates on the line from (x0,y0) to (x1,y1), inclusive."""
    res = []
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1
    dx = abs(dx)
    dy = abs(dy)
    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0
    D = 2 * dy - dx
    y = 0
    for x in range(dx + 1):
        res.append((x0 + x * xx + y * yx, y0 + x * xy + y * yy))
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy
    return res


# ---------------------------------------------------------------------------
# Quadtree node
# ---------------------------------------------------------------------------

class _QuadNode:
    """Internal quadtree node.  Leaf when children is None."""
    __slots__ = ("cx", "cy", "half", "value", "children")

    def __init__(self, cx: float, cy: float, half: float, value: int = UNKNOWN):
        self.cx = cx          # centre x in world coords
        self.cy = cy          # centre y
        self.half = half      # half-width of this node's extent
        self.value = value    # cell value (only meaningful for leaves)
        self.children = None  # None → leaf, else tuple(nw, ne, sw, se)

    def is_leaf(self) -> bool:
        return self.children is None

    def subdivide(self):
        h = self.half / 2.0
        v = self.value
        self.children = (
            _QuadNode(self.cx - h, self.cy + h, h, v),  # NW
            _QuadNode(self.cx + h, self.cy + h, h, v),  # NE
            _QuadNode(self.cx - h, self.cy - h, h, v),  # SW
            _QuadNode(self.cx + h, self.cy - h, h, v),  # SE
        )
        self.value = UNKNOWN  # interior nodes have no single value

    def _child_index(self, wx: float, wy: float) -> int:
        """Return 0-3 index of the child quadrant containing (wx, wy)."""
        xi = 0 if wx < self.cx else 1
        yi = 0 if wy >= self.cy else 1  # NW/NE = row 0, SW/SE = row 1
        return yi * 2 + xi

    def try_merge(self):
        """If all children are leaves with the same value, collapse back."""
        if self.children is None:
            return
        vals = set()
        for c in self.children:
            if not c.is_leaf():
                return
            vals.add(c.value)
        if len(vals) == 1:
            self.value = vals.pop()
            self.children = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class QuadtreeOccupancyGrid:
    """Quadtree-backed occupancy grid with lidar integration, frontier
    detection, obstacle inflation, and A* path planning."""

    def __init__(
        self,
        world_width: float,
        world_height: float,
        min_resolution: float = 1.0,
        origin_x: float | None = None,
        origin_y: float | None = None,
    ):
        # World is centred at (0,0) by default
        self.min_resolution = min_resolution
        self.world_width = world_width
        self.world_height = world_height

        # Origin = bottom-left corner of the grid (min x, min y)
        self.origin_x = origin_x if origin_x is not None else -world_width / 2.0
        self.origin_y = origin_y if origin_y is not None else -world_height / 2.0

        # Quadtree covers a square whose side = max(width, height), power-of-2
        side = max(world_width, world_height)
        # Round up to next power of 2 of min_resolution multiples
        n_cells = int(math.ceil(side / min_resolution))
        pot = 1
        while pot < n_cells:
            pot <<= 1
        self._grid_side = pot * min_resolution  # world-space side length

        cx = self.origin_x + self._grid_side / 2.0
        cy = self.origin_y + self._grid_side / 2.0
        self._root = _QuadNode(cx, cy, self._grid_side / 2.0, UNKNOWN)

        # Number of cells along each axis at finest resolution
        self.cells_x = int(math.ceil(world_width / min_resolution))
        self.cells_y = int(math.ceil(world_height / min_resolution))

        # Cached flattened grid (invalidated on writes)
        self._flat_cache: Optional[np.ndarray] = None
        self._inflated_cache: Optional[np.ndarray] = None
        self._inflated_radius: float = -1.0

    # ---- coordinate helpers -----------------------------------------------

    def world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coords to cell indices (col, row)."""
        col = int((wx - self.origin_x) / self.min_resolution)
        row = int((wy - self.origin_y) / self.min_resolution)
        return col, row

    def cell_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert cell indices to world coords (centre of cell)."""
        wx = self.origin_x + (col + 0.5) * self.min_resolution
        wy = self.origin_y + (row + 0.5) * self.min_resolution
        return wx, wy

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cells_x and 0 <= row < self.cells_y

    # ---- get / set --------------------------------------------------------

    def get_cell(self, wx: float, wy: float) -> int:
        node = self._root
        while not node.is_leaf():
            idx = node._child_index(wx, wy)
            node = node.children[idx]
        return node.value

    def set_cell(self, wx: float, wy: float, value: int):
        self._flat_cache = None
        self._inflated_cache = None
        self._set_recursive(self._root, wx, wy, value)

    def _set_recursive(self, node: _QuadNode, wx: float, wy: float, value: int):
        if node.is_leaf():
            if node.half <= self.min_resolution:
                node.value = value
                return
            if node.value == value:
                return  # already this value everywhere
            node.subdivide()
        idx = node._child_index(wx, wy)
        self._set_recursive(node.children[idx], wx, wy, value)
        node.try_merge()

    # ---- lidar integration ------------------------------------------------

    def update_from_lidar(
        self,
        agent_x: float,
        agent_y: float,
        agent_yaw: float,
        ranges: list | np.ndarray,
        angle_start_rad: float,
        angle_increment_rad: float,
        max_range: float,
    ):
        """Integrate a 2D lidar scan into the occupancy grid.

        Rays are cast from the agent position.  Free cells are marked along
        each ray; the endpoint is marked occupied if the range < max_range.
        """
        self._flat_cache = None
        self._inflated_cache = None

        cos_yaw = math.cos(agent_yaw)
        sin_yaw = math.sin(agent_yaw)
        agent_col, agent_row = self.world_to_cell(agent_x, agent_y)

        for i, r in enumerate(ranges):
            angle_local = angle_start_rad + i * angle_increment_rad
            # Local frame: x=forward, y=left → world rotation
            dx_local = math.cos(angle_local)
            dy_local = math.sin(angle_local)
            dx_world = cos_yaw * dx_local - sin_yaw * dy_local
            dy_world = sin_yaw * dx_local + cos_yaw * dy_local

            hit = r > 0 and r < max_range
            ray_len = r if hit else max_range

            end_x = agent_x + dx_world * ray_len
            end_y = agent_y + dy_world * ray_len
            end_col, end_row = self.world_to_cell(end_x, end_y)

            # Bresenham from agent to endpoint
            cells = bresenham(agent_col, agent_row, end_col, end_row)
            # Mark all but last cell as free
            for cx, cy in cells[:-1]:
                if self._in_bounds(cx, cy):
                    wx, wy = self.cell_to_world(cx, cy)
                    cur = self.get_cell(wx, wy)
                    if cur != OCCUPIED:  # don't overwrite occupied cells
                        self._set_cell_fast(wx, wy, FREE)

            # Mark endpoint
            if hit and self._in_bounds(end_col, end_row):
                wx, wy = self.cell_to_world(end_col, end_row)
                self._set_cell_fast(wx, wy, OCCUPIED)
            elif self._in_bounds(end_col, end_row):
                # Max range reached — mark as free
                wx, wy = self.cell_to_world(end_col, end_row)
                cur = self.get_cell(wx, wy)
                if cur != OCCUPIED:
                    self._set_cell_fast(wx, wy, FREE)

    def _set_cell_fast(self, wx: float, wy: float, value: int):
        """Set cell without invalidating cache (caller handles it)."""
        self._set_recursive(self._root, wx, wy, value)

    # ---- flatten to numpy -------------------------------------------------

    def to_flat_grid(self) -> np.ndarray:
        """Return (cells_y, cells_x) int8 array with cell values."""
        if self._flat_cache is not None:
            return self._flat_cache
        grid = np.full((self.cells_y, self.cells_x), UNKNOWN, dtype=np.int8)
        self._flatten_node(self._root, grid)
        self._flat_cache = grid
        return grid

    def _flatten_node(self, node: _QuadNode, grid: np.ndarray):
        if node.is_leaf():
            # Compute cell range covered by this node
            min_wx = node.cx - node.half
            max_wx = node.cx + node.half
            min_wy = node.cy - node.half
            max_wy = node.cy + node.half
            col0 = max(0, int((min_wx - self.origin_x) / self.min_resolution))
            col1 = min(self.cells_x, int(math.ceil((max_wx - self.origin_x) / self.min_resolution)))
            row0 = max(0, int((min_wy - self.origin_y) / self.min_resolution))
            row1 = min(self.cells_y, int(math.ceil((max_wy - self.origin_y) / self.min_resolution)))
            if col0 < col1 and row0 < row1:
                grid[row0:row1, col0:col1] = node.value
        else:
            for child in node.children:
                self._flatten_node(child, grid)

    # ---- obstacle inflation -----------------------------------------------

    def get_inflated_grid(self, inflation_radius: float) -> np.ndarray:
        """Return grid with occupied cells dilated by inflation_radius.

        Result values: UNKNOWN=-1, FREE=0, OCCUPIED=100.
        """
        if (self._inflated_cache is not None
                and self._inflated_radius == inflation_radius):
            return self._inflated_cache

        grid = self.to_flat_grid().copy()
        if inflation_radius <= 0:
            self._inflated_cache = grid
            self._inflated_radius = inflation_radius
            return grid

        r_cells = int(math.ceil(inflation_radius / self.min_resolution))
        occupied_mask = (grid == OCCUPIED)

        # Build circular structuring element
        y_offsets, x_offsets = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
        kernel = (x_offsets ** 2 + y_offsets ** 2) <= r_cells ** 2

        # Dilate using convolution-like approach
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(occupied_mask, structure=kernel)

        # Apply: mark newly inflated cells as occupied, but don't touch unknown
        inflated_grid = grid.copy()
        new_occupied = dilated & (~occupied_mask) & (grid != UNKNOWN)
        inflated_grid[new_occupied] = OCCUPIED

        self._inflated_cache = inflated_grid
        self._inflated_radius = inflation_radius
        return inflated_grid

    # ---- frontier detection -----------------------------------------------

    def get_frontier_cells(self) -> List[Tuple[int, int]]:
        """Return list of (col, row) cells that are FREE and adjacent to UNKNOWN."""
        grid = self.to_flat_grid()
        free_mask = (grid == FREE)
        unknown_mask = (grid == UNKNOWN)

        # Pad unknown mask to handle edges
        padded = np.pad(unknown_mask, 1, mode="constant", constant_values=False)

        # Check 4-connected neighbors for unknown
        has_unknown_neighbor = (
            padded[:-2, 1:-1] |   # up
            padded[2:, 1:-1] |    # down
            padded[1:-1, :-2] |   # left
            padded[1:-1, 2:]      # right
        )

        frontier_mask = free_mask & has_unknown_neighbor
        rows, cols = np.nonzero(frontier_mask)
        return list(zip(cols.tolist(), rows.tolist()))

    def cluster_frontiers(
        self, frontier_cells: List[Tuple[int, int]], min_size: int = 5
    ) -> List[List[Tuple[int, int]]]:
        """Cluster frontier cells via flood-fill. Return clusters >= min_size."""
        if not frontier_cells:
            return []

        cell_set = set(frontier_cells)
        visited = set()
        clusters = []

        for cell in frontier_cells:
            if cell in visited:
                continue
            # BFS
            cluster = []
            queue = [cell]
            visited.add(cell)
            while queue:
                c = queue.pop()
                cluster.append(c)
                cx, cy = c
                for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                    n = (nx, ny)
                    if n in cell_set and n not in visited:
                        visited.add(n)
                        queue.append(n)
            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    @staticmethod
    def cluster_centroid(cluster: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Return (col, row) centroid of a cluster."""
        cols = [c[0] for c in cluster]
        rows = [c[1] for c in cluster]
        return sum(cols) / len(cols), sum(rows) / len(rows)

    # ---- A* path planning -------------------------------------------------

    def astar(
        self,
        start_wx: float,
        start_wy: float,
        goal_wx: float,
        goal_wy: float,
        inflation_radius: float = 2.0,
    ) -> Optional[List[Tuple[float, float]]]:
        """A* on the inflated grid. Returns list of (wx, wy) waypoints or None.

        If the start cell is inside the inflated zone (robot pushed into wall),
        the search is allowed to expand from occupied cells near the start so
        the robot can escape.
        """
        grid = self.get_inflated_grid(inflation_radius)
        sc, sr = self.world_to_cell(start_wx, start_wy)
        gc, gr = self.world_to_cell(goal_wx, goal_wy)

        if not self._in_bounds(sc, sr) or not self._in_bounds(gc, gr):
            return None
        if grid[gr, gc] == OCCUPIED:
            return None

        # If start is in inflated zone, allow escaping: mark cells within a
        # small radius of the start as passable.
        start_in_obstacle = grid[sr, sc] == OCCUPIED
        escape_set = set()
        if start_in_obstacle:
            escape_r = int(math.ceil(inflation_radius / self.min_resolution)) + 1
            for dr in range(-escape_r, escape_r + 1):
                for dc in range(-escape_r, escape_r + 1):
                    nc, nr = sc + dc, sr + dr
                    if self._in_bounds(nc, nr) and dc * dc + dr * dr <= escape_r * escape_r:
                        escape_set.add((nc, nr))

        # 8-connected neighbors with costs
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414),
        ]

        def heuristic(c, r):
            return math.sqrt((c - gc) ** 2 + (r - gr) ** 2)

        open_set = [(heuristic(sc, sr), 0.0, sc, sr)]
        g_score = {(sc, sr): 0.0}
        came_from = {}

        while open_set:
            f, g, cc, cr = heapq.heappop(open_set)

            if cc == gc and cr == gr:
                # Reconstruct path
                path = []
                node = (gc, gr)
                while node in came_from:
                    wx, wy = self.cell_to_world(node[0], node[1])
                    path.append((wx, wy))
                    node = came_from[node]
                wx, wy = self.cell_to_world(sc, sr)
                path.append((wx, wy))
                path.reverse()
                return path

            if g > g_score.get((cc, cr), float("inf")):
                continue

            for dc, dr, cost in neighbors:
                nc, nr = cc + dc, cr + dr
                if not self._in_bounds(nc, nr):
                    continue
                cell_val = grid[nr, nc]
                if cell_val == OCCUPIED and (nc, nr) not in escape_set:
                    continue
                # Allow traversal through unknown — we'll replan if we discover obstacles
                # Add penalty for traversing inflated cells to prefer clear paths
                extra_cost = 5.0 if cell_val == OCCUPIED else 0.0
                ng = g + cost + extra_cost
                if ng < g_score.get((nc, nr), float("inf")):
                    g_score[(nc, nr)] = ng
                    came_from[(nc, nr)] = (cc, cr)
                    heapq.heappush(open_set, (ng + heuristic(nc, nr), ng, nc, nr))

        return None  # no path found

    # ---- OccupancyGrid message export -------------------------------------

    def to_occupancy_grid_msg(self, frame_id: str = "odom", stamp=None):
        """Build a nav_msgs/OccupancyGrid message from the current state."""
        from nav_msgs.msg import OccupancyGrid, MapMetaData
        from std_msgs.msg import Header
        from geometry_msgs.msg import Pose, Point, Quaternion

        grid = self.to_flat_grid()

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.frame_id = frame_id
        if stamp is not None:
            msg.header.stamp = stamp

        msg.info = MapMetaData()
        msg.info.resolution = self.min_resolution
        msg.info.width = self.cells_x
        msg.info.height = self.cells_y
        msg.info.origin = Pose()
        msg.info.origin.position = Point(
            x=self.origin_x, y=self.origin_y, z=0.0
        )
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Flatten row-major (ROS convention: row 0 = bottom)
        msg.data = grid.flatten().astype(np.int8).tolist()

        return msg

    # ---- reset ------------------------------------------------------------

    def clear(self):
        """Reset the entire grid to UNKNOWN."""
        cx = self.origin_x + self._grid_side / 2.0
        cy = self.origin_y + self._grid_side / 2.0
        self._root = _QuadNode(cx, cy, self._grid_side / 2.0, UNKNOWN)
        self._flat_cache = None
        self._inflated_cache = None
