"""Waypoint-based trajectory follower for Quadruped-PyMPC.

Produces (cmd_vx, cmd_vy, cmd_wz) reference velocity commands given a list of
(x, y, yaw) waypoints and the robot's current pose. The output replaces the
keyboard-driven ref_base_lin_vel / ref_base_ang_vel that simulation.py would
otherwise read from env.target_base_vel().

Conventions match the rest of Quadruped-PyMPC:
    - lin_vel is a 3-vector in WORLD frame (the MPC handles body-frame conversion)
    - ang_vel is a 3-vector with yaw rate in component [2]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Waypoint:
    """A single 2D waypoint with a target heading."""
    x: float
    y: float
    yaw: float            # radians
    hold_time: float = 0.0  # seconds to hold at this waypoint before advancing


@dataclass
class FollowerConfig:
    """Tunable parameters for the waypoint follower.

    Defaults are tuned for a real walking quadruped (Aliengo / Mini Cheetah)
    using Quadruped-PyMPC's trot gait. Velocity ranges match what the MPC
    can actually realize.
    """
    # P-controller gains (command per unit error)
    k_v: float = 1.2            # m/s per m of position error
    k_w: float = 1.5            # rad/s per rad of yaw error

    # Saturation limits — match the MPC's trained range
    v_max: float = 0.2         # m/s — comfortable trot speed
    w_max: float = 0.4          # rad/s

    # Tolerances for "waypoint reached"
    pos_tol: float = 0.20       # m
    yaw_tol: float = np.deg2rad(15.0)  # rad

    # Per-waypoint timeout so the robot doesn't get stuck
    waypoint_timeout: float = 30.0  # seconds

    # Slow down when this close to the target (avoids overshoot)
    decel_radius: float = 0.6   # m


class WaypointFollower:
    """Stateful waypoint follower.

    Per-step usage in simulation_waypoints.py:
        ref_lin, ref_ang = follower.step(base_pos, yaw, t)
        # ref_lin: np.ndarray shape (3,) in world frame
        # ref_ang: np.ndarray shape (3,) with yaw rate in [2]
    """

    def __init__(self, waypoints: List[Waypoint],
                 config: Optional[FollowerConfig] = None):
        if not waypoints:
            raise ValueError("waypoints list cannot be empty")
        self.waypoints = waypoints
        self.cfg = config or FollowerConfig()

        # State
        self.current_idx = 0
        self.wp_start_time: Optional[float] = None
        self.hold_start_time: Optional[float] = None
        self.finished = False

        # Logs (used by the runner for plotting and metrics)
        self.log_t: List[float] = []
        self.log_target_idx: List[int] = []
        self.log_cross_track: List[float] = []
        self.log_heading_err: List[float] = []
        self.log_dist_to_wp: List[float] = []

    # ─────────────────────── public API ───────────────────────────────

    def step(self, base_pos: np.ndarray, yaw: float, t: float
             ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reference velocities for this control tick.

        Args:
            base_pos: shape (3,) world-frame position [x, y, z]
            yaw: scalar yaw angle (rad)
            t: current simulation time (s)

        Returns:
            (ref_lin_vel, ref_ang_vel) — both shape (3,), in WORLD frame.
            yaw rate is in ref_ang_vel[2].
        """
        if self.finished:
            return np.zeros(3), np.zeros(3)

        if self.wp_start_time is None:
            self.wp_start_time = t

        target = self.waypoints[self.current_idx]
        x, y = float(base_pos[0]), float(base_pos[1])

        # ---- World-frame errors ----
        dx_w = target.x - x
        dy_w = target.y - y
        dist = float(np.hypot(dx_w, dy_w))
        yaw_err = _wrap_to_pi(target.yaw - yaw)

        # ---- Waypoint reached? ----
        reached = (dist < self.cfg.pos_tol) and (abs(yaw_err) < self.cfg.yaw_tol)
        if reached:
            if self.hold_start_time is None:
                self.hold_start_time = t
            if (t - self.hold_start_time) >= target.hold_time:
                self._advance(t)
                if self.finished:
                    self._log(t, 0.0, 0.0, 0.0)
                    return np.zeros(3), np.zeros(3)
                return self.step(base_pos, yaw, t)  # recurse for new target
            self._log(t, dist, yaw_err, 0.0)
            return np.zeros(3), np.zeros(3)
        else:
            self.hold_start_time = None

        # ---- Timeout ----
        if (t - self.wp_start_time) > self.cfg.waypoint_timeout:
            print(f"  [trajectory] waypoint {self.current_idx} timed out after "
                  f"{self.cfg.waypoint_timeout:.1f}s at dist={dist:.2f}m -- skipping")
            self._advance(t)
            return self.step(base_pos, yaw, t)

        # ---- Decelerate near the target ----
        scale = 1.0
        if dist < self.cfg.decel_radius:
            scale = max(dist / self.cfg.decel_radius, 0.25)

        # ---- Velocity command in WORLD frame (MPC expects world frame) ----
        # Direction toward the target, scaled by P-gain and saturated
        vx_w = np.clip(self.cfg.k_v * dx_w * scale, -self.cfg.v_max, self.cfg.v_max)
        vy_w = np.clip(self.cfg.k_v * dy_w * scale, -self.cfg.v_max, self.cfg.v_max)
        wz   = np.clip(self.cfg.k_w * yaw_err,      -self.cfg.w_max, self.cfg.w_max)

        ref_lin_vel = np.array([vx_w, vy_w, 0.0])
        ref_ang_vel = np.array([0.0, 0.0, wz])

        # ---- Cross-track error (for metrics) ----
        cross_track = self._cross_track_error(x, y)
        self._log(t, dist, yaw_err, cross_track)

        return ref_lin_vel, ref_ang_vel

    def is_finished(self) -> bool:
        return self.finished

    def current_target(self) -> Optional[Waypoint]:
        if self.finished:
            return None
        return self.waypoints[self.current_idx]

    def progress(self) -> Tuple[int, int]:
        return self.current_idx, len(self.waypoints)

    # ─────────────────────── internals ────────────────────────────────

    def _advance(self, t: float):
        self.current_idx += 1
        self.wp_start_time = t
        self.hold_start_time = None
        if self.current_idx >= len(self.waypoints):
            self.finished = True

    def _cross_track_error(self, x: float, y: float) -> float:
        """Perpendicular distance from (x, y) to the segment prev_wp -> current_wp."""
        if self.current_idx == 0:
            x0, y0 = 0.0, 0.0
        else:
            prev = self.waypoints[self.current_idx - 1]
            x0, y0 = prev.x, prev.y
        tgt = self.waypoints[self.current_idx]
        x1, y1 = tgt.x, tgt.y

        sx, sy = x1 - x0, y1 - y0
        seg_len_sq = sx * sx + sy * sy
        if seg_len_sq < 1e-9:
            return float(np.hypot(x - x0, y - y0))

        tproj = ((x - x0) * sx + (y - y0) * sy) / seg_len_sq
        tproj = max(0.0, min(1.0, tproj))
        px = x0 + tproj * sx
        py = y0 + tproj * sy
        return float(np.hypot(x - px, y - py))

    def _log(self, t: float, dist: float, yaw_err: float, cross_track: float):
        self.log_t.append(t)
        self.log_target_idx.append(self.current_idx)
        self.log_dist_to_wp.append(dist)
        self.log_heading_err.append(yaw_err)
        self.log_cross_track.append(cross_track)


# ─────────────────────── helpers ──────────────────────────────────────

def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def summarize_tracking(follower: WaypointFollower) -> dict:
    """Compute scalar tracking metrics for reporting."""
    if not follower.log_cross_track:
        return {}
    ct = np.asarray(follower.log_cross_track)
    he = np.asarray(follower.log_heading_err)
    return {
        "cross_track_rms_m":     float(np.sqrt(np.mean(ct ** 2))),
        "cross_track_max_m":     float(np.max(np.abs(ct))),
        "heading_err_rms_deg":   float(np.sqrt(np.mean(he ** 2)) * 180.0 / np.pi),
        "heading_err_max_deg":   float(np.max(np.abs(he)) * 180.0 / np.pi),
        "waypoints_reached":     follower.current_idx,
        "waypoints_total":       len(follower.waypoints),
        "completed":             follower.finished,
    }
