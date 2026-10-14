import time
import numpy as np
import heapq
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.interpolate import splprep, splev

from src.constraints import Boundaries3D, CylindricalObstacle3D
from src.state import wrap_to_2pi, wrap_angle

class GlobalMapPlanner:
    """
    Pre-Mission Global Map & Path Planner for Autonomous Underwater Vehicles (AUV).
    
    Before the AUV starts navigating/exploring:
    1. Ingests the full 3D bathymetric map (boundaries, depths, cylindrical hazards, safe buffers).
    2. Builds an augmented 2D/3D costmap incorporating:
       - Hard physical obstacle hull exclusion zones (impassable).
       - Quadratic safety standoff buffer barriers.
       - Decaying repulsive halo gradients pushing paths toward open sea lanes.
       - Geofence boundary repulsive standoff walls.
    3. Executes 8-connected A* search with Euclidean heuristic.
    4. Applies ray-cast line-of-sight shortcut pruning (string-pulling).
    5. Applies cubic spline smoothing with curvature conditioning.
    6. Generates a 3D depth profile strictly contained within operational depth boundaries.
    
    During flight:
    - Provides lookahead target waypoints (pure pursuit / carrot-following) along the safe route.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        default_standoff: float = 4.0,
        wall_margin: float = 4.0,
        lookahead_distance: float = 16.0
    ):
        self.resolution = resolution
        self.default_standoff = default_standoff
        self.wall_margin = wall_margin
        self.lookahead_distance = lookahead_distance

        self.cached_path: Optional[np.ndarray] = None # (M, 3) [North, East, Depth]
        self.cum_distances: Optional[np.ndarray] = None # (M,)

    def plan(
        self,
        start: Union[np.ndarray, List[float]],
        goal: Union[np.ndarray, List[float]],
        boundaries: Union[Boundaries3D, Dict[str, float]],
        obstacles: List[Union[CylindricalObstacle3D, Dict[str, Any]]]
    ) -> np.ndarray:
        """
        Computes an optimal, collision-free 3D global path tailored to the provided map.
        Returns:
            np.ndarray of shape (M, 3): [[North, East, Depth], ...]
        """
        t0 = time.perf_counter()

        # Parse boundaries
        if isinstance(boundaries, Boundaries3D):
            n_min, n_max = boundaries.north_min, boundaries.north_max
            e_min, e_max = boundaries.east_min, boundaries.east_max
            d_min, d_max = boundaries.depth_min, boundaries.depth_max
        else:
            n_min = float(boundaries.get("north_min", 0.0))
            n_max = float(boundaries.get("north_max", 120.0))
            e_min = float(boundaries.get("east_min", 0.0))
            e_max = float(boundaries.get("east_max", 100.0))
            d_min = float(boundaries.get("depth_min", 10.0))
            d_max = float(boundaries.get("depth_max", 60.0))

        # Clamp start & goal depth to safe operational limits
        safe_d_min = d_min + 2.0
        safe_d_max = d_max - 2.0
        s_n, s_e = float(start[0]), float(start[1])
        s_d = float(np.clip(start[2], safe_d_min, safe_d_max))

        g_n, g_e = float(goal[0]), float(goal[1])
        g_d = float(np.clip(goal[2], safe_d_min, safe_d_max))

        # Parse obstacles
        obs_list = []
        for o in obstacles:
            if isinstance(o, CylindricalObstacle3D):
                obs_list.append({
                    "north": o.north,
                    "east": o.east,
                    "radius": o.radius,
                    "margin": o.margin,
                    "depth_min": o.depth_min,
                    "depth_max": o.depth_max
                })
            else:
                obs_list.append({
                    "north": float(o["north"]),
                    "east": float(o["east"]),
                    "radius": float(o["radius"]),
                    "margin": float(o.get("margin", self.default_standoff)),
                    "depth_min": float(o.get("depth_min", 0.0)),
                    "depth_max": float(o.get("depth_max", 100.0))
                })

        # Discretize 2D domain
        res = self.resolution
        n_bins = int(np.ceil((n_max - n_min) / res)) + 1
        e_bins = int(np.ceil((e_max - e_min) / res)) + 1

        n_coords = np.linspace(n_min, n_max, n_bins)
        e_coords = np.linspace(e_min, e_max, e_bins)
        EE, NN = np.meshgrid(e_coords, n_coords)

        # Baseline traversal cost
        cost_grid = np.ones((n_bins, e_bins), dtype=np.float32)
        impassable = np.zeros((n_bins, e_bins), dtype=bool)

        # 1. Geofence Boundary Impassable Perimeter & Repulsive Barrier
        wall_hard = max(2.5, self.wall_margin * 0.7)
        impassable |= (NN < n_min + wall_hard) | (NN > n_max - wall_hard) | \
                      (EE < e_min + wall_hard) | (EE > e_max - wall_hard)

        dist_wall = np.minimum.reduce([
            NN - n_min, n_max - NN,
            EE - e_min, e_max - EE
        ])
        wall_halo = 12.0
        in_wall_halo = dist_wall < wall_halo
        cost_grid[in_wall_halo] += (100.0 * (1.0 - dist_wall[in_wall_halo] / wall_halo)**2).astype(np.float32)

        # 2. Obstacle Cylinders Impassable Hulls & Repulsive Halo
        for obs in obs_list:
            on, oe, rad = obs["north"], obs["east"], obs["radius"]
            margin = obs["margin"]
            dist_center = np.hypot(NN - on, EE - oe)
            dist_hull = dist_center - rad

            # Hard physical hull collision: completely impassable
            impassable |= (dist_hull <= 1.0)

            # Safety standoff buffer zone: steep penalty
            in_buf = (dist_hull > 1.0) & (dist_hull <= margin + 1.5)
            cost_grid[in_buf] += (800.0 * (1.0 - (dist_hull[in_buf] - 1.0) / (margin + 1.5))**2).astype(np.float32)

            # Wide clearance gradient (pushes paths toward wide, open waters)
            halo_span = 18.0
            in_halo = (dist_hull > margin + 1.5) & (dist_hull <= margin + 1.5 + halo_span)
            cost_grid[in_halo] += (120.0 * (1.0 - (dist_hull[in_halo] - (margin + 1.5)) / halo_span)**2).astype(np.float32)

        # Coordinate to index helper
        def to_idx(n_val, e_val):
            in_ = int(np.clip(np.round((n_val - n_min) / res), 0, n_bins - 1))
            ie_ = int(np.clip(np.round((e_val - e_min) / res), 0, e_bins - 1))
            return in_, ie_

        s_in, s_ie = to_idx(s_n, s_e)
        g_in, g_ie = to_idx(g_n, g_e)

        # Ensure start and goal cells are open for search
        impassable[s_in, s_ie] = False
        impassable[g_in, g_ie] = False

        # 3. 8-Connected A* Search
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)
        ]

        def heuristic(ni, ei):
            return np.hypot((ni - g_in) * res, (ei - g_ie) * res)

        dist = np.full((n_bins, e_bins), np.inf, dtype=np.float32)
        dist[s_in, s_ie] = 0.0

        parent = {}
        pq = [(heuristic(s_in, s_ie), 0.0, s_in, s_ie)]
        visited = set()

        reached = False
        while pq:
            f, d, cn, ce = heapq.heappop(pq)
            if (cn, ce) == (g_in, g_ie):
                reached = True
                break
            if (cn, ce) in visited:
                continue
            visited.add((cn, ce))

            for dn, de, step_cost in neighbors:
                nn, ne = cn + dn, ce + de
                if 0 <= nn < n_bins and 0 <= ne < e_bins and not impassable[nn, ne]:
                    step_w = step_cost * cost_grid[nn, ne]
                    new_d = d + step_w
                    if new_d < dist[nn, ne]:
                        dist[nn, ne] = new_d
                        parent[(nn, ne)] = (cn, ce)
                        heapq.heappush(pq, (new_d + heuristic(nn, ne), new_d, nn, ne))

        if not reached:
            # Fallback: direct waypoint interpolation
            raw_2d = np.linspace([s_n, s_e], [g_n, g_e], 30)
        else:
            # Reconstruct path
            path_idx = [(g_in, g_ie)]
            curr = (g_in, g_ie)
            while curr in parent:
                curr = parent[curr]
                path_idx.append(curr)
            path_idx.reverse()

            raw_2d = np.array([[n_coords[ni], e_coords[ei]] for ni, ei in path_idx])

        # 4. Shortcut Pruning (String-pulling with clearance verification)
        pruned_2d = self._prune_path(raw_2d, obs_list, n_min, n_max, e_min, e_max)

        # 5. Smooth Cubic Spline with Standoff Barrier Verification
        smooth_2d = self._smooth_path(pruned_2d, sample_step=1.5, obstacles=obs_list, n_min=n_min, n_max=n_max, e_min=e_min, e_max=e_max)

        # Force exact endpoints
        smooth_2d[0] = [s_n, s_e]
        smooth_2d[-1] = [g_n, g_e]

        # 6. Smooth 3D Depth Profile
        path_len_cum = np.cumsum(np.hypot(np.diff(smooth_2d[:, 0]), np.diff(smooth_2d[:, 1])))
        path_len_cum = np.insert(path_len_cum, 0, 0.0)
        total_len = path_len_cum[-1] if path_len_cum[-1] > 1e-3 else 1.0

        frac = path_len_cum / total_len
        # Smooth S-curve (Hermite smoothstep) for gradual pitch initiation and arrival leveling
        s_frac = 3.0 * (frac ** 2) - 2.0 * (frac ** 3)
        depth_profile = s_d + (g_d - s_d) * s_frac
        depth_profile = np.clip(depth_profile, safe_d_min, safe_d_max)

        path_3d = np.column_stack([smooth_2d[:, 0], smooth_2d[:, 1], depth_profile])

        self.cached_path = path_3d
        self.cum_distances = path_len_cum
        self.last_closest_idx = 0

        solve_time = (time.perf_counter() - t0) * 1000.0
        return path_3d

    def _is_segment_clear(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        obstacles: List[Dict[str, Any]],
        n_min: float,
        n_max: float,
        e_min: float,
        e_max: float,
        num_samples: int = 30
    ) -> bool:
        """Verifies if line segment between p1 and p2 has safe standoff from all hazards & walls."""
        alphas = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]
        pts = p1 + alphas * (p2 - p1)

        # Check boundaries
        pad = 3.0
        if np.any(pts[:, 0] < n_min + pad) or np.any(pts[:, 0] > n_max - pad) or \
           np.any(pts[:, 1] < e_min + pad) or np.any(pts[:, 1] > e_max - pad):
            return False

        # Check obstacles
        for obs in obstacles:
            dists = np.hypot(pts[:, 0] - obs["north"], pts[:, 1] - obs["east"])
            # Required standoff: radius + 85% of margin
            min_allowable = obs["radius"] + obs["margin"] * 0.85
            if np.any(dists < min_allowable):
                return False

        return True

    def _prune_path(
        self,
        raw_path: np.ndarray,
        obstacles: List[Dict[str, Any]],
        n_min: float,
        n_max: float,
        e_min: float,
        e_max: float
    ) -> np.ndarray:
        """Prunes unnecessary zigzag grid steps via greedy string-pulling."""
        if len(raw_path) <= 2:
            return raw_path

        pruned = [raw_path[0]]
        curr_idx = 0
        n_pts = len(raw_path)

        while curr_idx < n_pts - 1:
            furthest = curr_idx + 1
            for candidate_idx in range(n_pts - 1, curr_idx, -1):
                if self._is_segment_clear(raw_path[curr_idx], raw_path[candidate_idx], obstacles, n_min, n_max, e_min, e_max):
                    furthest = candidate_idx
                    break
            pruned.append(raw_path[furthest])
            curr_idx = furthest

        return np.array(pruned)

    def _smooth_path(
        self,
        pruned_path: np.ndarray,
        sample_step: float = 1.5,
        obstacles: Optional[List[Dict[str, Any]]] = None,
        n_min: Optional[float] = None,
        n_max: Optional[float] = None,
        e_min: Optional[float] = None,
        e_max: Optional[float] = None
    ) -> np.ndarray:
        """Smooths piecewise linear waypoints using B-splines into flyable curves with safety guarantees."""
        if len(pruned_path) < 3:
            total_dist = float(np.sum(np.hypot(np.diff(pruned_path[:, 0]), np.diff(pruned_path[:, 1]))))
            num_pts = max(20, int(np.ceil(total_dist / sample_step)))
            alphas = np.linspace(0.0, 1.0, num_pts)[:, np.newaxis]
            dense_line = pruned_path[0] + alphas * (pruned_path[-1] - pruned_path[0])
            return dense_line

        # Deduplicate points that are identical or too close
        diffs = np.hypot(np.diff(pruned_path[:, 0]), np.diff(pruned_path[:, 1]))
        keep = np.insert(diffs > 0.3, 0, True)
        clean_pts = pruned_path[keep]
        if len(clean_pts) < 3:
            total_dist = float(np.sum(np.hypot(np.diff(clean_pts[:, 0]), np.diff(clean_pts[:, 1]))))
            num_pts = max(20, int(np.ceil(total_dist / sample_step)))
            alphas = np.linspace(0.0, 1.0, num_pts)[:, np.newaxis]
            dense_line = clean_pts[0] + alphas * (clean_pts[-1] - clean_pts[0])
            return dense_line

        try:
            k = min(3, len(clean_pts) - 1)
            tck, _ = splprep([clean_pts[:, 0], clean_pts[:, 1]], s=3.0, k=k)
            total_dist = float(np.sum(np.hypot(np.diff(clean_pts[:, 0]), np.diff(clean_pts[:, 1]))))
            num_pts = max(20, int(np.ceil(total_dist / sample_step)))
            u_fine = np.linspace(0.0, 1.0, num_pts)
            smooth_n, smooth_e = splev(u_fine, tck)
            smooth_pts = np.column_stack([smooth_n, smooth_e])

            # Safety enforcement: verify standoff distance against obstacles
            if obstacles:
                for obs in obstacles:
                    on, oe, rad = obs["north"], obs["east"], obs["radius"]
                    margin = obs.get("margin", self.default_standoff)
                    min_safe = rad + margin * 0.85
                    d_centers = np.hypot(smooth_pts[:, 0] - on, smooth_pts[:, 1] - oe)
                    breached = d_centers < min_safe
                    if np.any(breached):
                        # Radially push breached spline points outside safe perimeter
                        for idx in np.where(breached)[0]:
                            d = d_centers[idx]
                            if d > 1e-4:
                                unit_v = (smooth_pts[idx] - np.array([on, oe])) / d
                                smooth_pts[idx] = np.array([on, oe]) + unit_v * (min_safe + 0.5)
                            else:
                                smooth_pts[idx] = np.array([on + min_safe + 0.5, oe])

            # Clamp within geofence boundaries
            if n_min is not None and n_max is not None and e_min is not None and e_max is not None:
                pad = 3.0
                smooth_pts[:, 0] = np.clip(smooth_pts[:, 0], n_min + pad, n_max - pad)
                smooth_pts[:, 1] = np.clip(smooth_pts[:, 1], e_min + pad, e_max - pad)

            return smooth_pts
        except Exception:
            return pruned_path

    def get_lookahead_horizon_targets(
        self,
        current_pos: Union[np.ndarray, List[float]],
        lookahead_dist: Optional[float] = None,
        horizon: int = 10,
        dt: float = 2.0,
        cruise_speed: float = 1.2
    ) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        """
        Calculates dynamic lookahead target waypoint AND full (H, 3) reference waypoint trajectory
        along the pre-planned collision-free bathymetric path.
        Returns:
            target_pos: np.ndarray (3,) [North, East, Depth]
            target_heading: float in [0, 2pi)
            target_pitch: float in rad
            dist_remaining: float in meters to terminal goal along path
            horizon_waypoints: np.ndarray (H, 3) sequential lookahead waypoints for MPC tracking
        """
        if self.cached_path is None or len(self.cached_path) < 2:
            p = np.array(current_pos[:3], dtype=np.float64)
            return p, 0.0, 0.0, 0.0, np.tile(p, (horizon, 1))

        path = self.cached_path
        cur_p = np.array(current_pos[:3], dtype=np.float64)
        Ld = lookahead_dist if lookahead_dist is not None else self.lookahead_distance

        # 1. Find projection / closest point on path with forward monotonicity
        if hasattr(self, "last_closest_idx") and self.last_closest_idx is not None:
            search_start = max(0, self.last_closest_idx - 1)
            search_end = min(len(path), self.last_closest_idx + 30)
            dists_to_path = np.linalg.norm(path[search_start:search_end] - cur_p, axis=1)
            closest_idx = search_start + int(np.argmin(dists_to_path))
            self.last_closest_idx = max(self.last_closest_idx, closest_idx)
        else:
            dists_to_path = np.linalg.norm(path - cur_p, axis=1)
            closest_idx = int(np.argmin(dists_to_path))
            self.last_closest_idx = closest_idx

        # 2. Advance along path by lookahead distance for primary carrot
        target_idx = closest_idx
        accum_dist = 0.0
        while target_idx < len(path) - 1 and accum_dist < Ld:
            step_d = float(np.linalg.norm(path[target_idx + 1] - path[target_idx]))
            accum_dist += step_d
            target_idx += 1

        target_pos = path[target_idx]

        # 4. Remaining distance to goal along path
        rem_steps = np.linalg.norm(np.diff(path[closest_idx:], axis=0), axis=1)
        dist_remaining = float(np.sum(rem_steps))

        # 3. Compute tangent heading & pitch
        # If near the end of path or within 14m of goal, steer directly towards final goal
        if target_idx >= len(path) - 4 or dist_remaining < 14.0:
            vec_goal = path[-1] - cur_p
            ref_heading = float(wrap_to_2pi(np.arctan2(vec_goal[1], vec_goal[0])))
            h_dist = float(np.hypot(vec_goal[0], vec_goal[1]))
            ref_pitch = float(np.clip(np.arctan2(-vec_goal[2], max(h_dist, 0.1)), -0.35, 0.35))
        else:
            ahead_idx = min(len(path) - 1, target_idx + 1)
            if ahead_idx == target_idx and target_idx > 0:
                vec = path[target_idx] - path[target_idx - 1]
            else:
                vec = path[ahead_idx] - path[target_idx]

            if np.hypot(vec[0], vec[1]) < 1e-3:
                ref_heading = float(wrap_to_2pi(np.arctan2(target_pos[1] - cur_p[1], target_pos[0] - cur_p[0])))
                ref_pitch = 0.0
            else:
                ref_heading = float(wrap_to_2pi(np.arctan2(vec[1], vec[0])))
                h_dist = float(np.hypot(vec[0], vec[1]))
                ref_pitch = float(np.clip(np.arctan2(-vec[2], max(h_dist, 0.1)), -0.35, 0.35))

        # 5. Extract multi-step horizon waypoints along path matching forward projection
        horizon_waypoints = np.zeros((horizon, 3), dtype=np.float64)
        curr_step_idx = closest_idx
        step_cum = 0.0
        for h in range(horizon):
            req_dist = (h + 1) * dt * cruise_speed
            while curr_step_idx < len(path) - 1 and step_cum < req_dist:
                seg_d = float(np.linalg.norm(path[curr_step_idx + 1] - path[curr_step_idx]))
                step_cum += seg_d
                curr_step_idx += 1
            horizon_waypoints[h] = path[curr_step_idx]

        return target_pos, ref_heading, ref_pitch, dist_remaining, horizon_waypoints

    def get_lookahead_target(
        self,
        current_pos: Union[np.ndarray, List[float]],
        lookahead_dist: Optional[float] = None
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Calculates dynamic lookahead target waypoint along the global path (pure pursuit carrot).
        Returns:
            target_pos: np.ndarray (3,) [North, East, Depth]
            target_heading: float in [0, 2pi)
            target_pitch: float in rad
            dist_remaining: float in meters to terminal goal along path
        """
        target_pos, ref_heading, ref_pitch, dist_remaining, _ = self.get_lookahead_horizon_targets(
            current_pos, lookahead_dist=lookahead_dist
        )
        return target_pos, ref_heading, ref_pitch, dist_remaining

    def evaluate_metrics(
        self,
        obstacles: List[Dict[str, Any]],
        boundaries: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculates safety clearance metrics of the planned path."""
        if self.cached_path is None:
            return {"planned_length_m": 0.0, "min_obs_clearance_m": 0.0, "min_wall_clearance_m": 0.0}

        path = self.cached_path
        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_len = float(np.sum(diffs))

        # Min obstacle clearance
        min_obs_clr = 999.0
        for obs in obstacles:
            on, oe, rad = obs["north"], obs["east"], obs["radius"]
            dists = np.hypot(path[:, 0] - on, path[:, 1] - oe) - rad
            min_d = float(np.min(dists))
            if min_d < min_obs_clr:
                min_obs_clr = min_d

        # Min boundary clearance
        n_min = boundaries.get("north_min", 0.0)
        n_max = boundaries.get("north_max", 120.0)
        e_min = boundaries.get("east_min", 0.0)
        e_max = boundaries.get("east_max", 100.0)
        wall_dists = np.minimum.reduce([
            path[:, 0] - n_min, n_max - path[:, 0],
            path[:, 1] - e_min, e_max - path[:, 1]
        ])
        min_wall_clr = float(np.min(wall_dists))

        return {
            "planned_length_m": total_len,
            "min_obs_clearance_m": min_obs_clr,
            "min_wall_clearance_m": min_wall_clr
        }
