"""Hardcoded waypoint trajectories for evaluating Quadruped-PyMPC.

Trajectories are sized for an actually-walking quadruped — meters of travel,
not centimeters. Each generator returns a list[Waypoint].

The robot starts at (0, 0, 0) facing +x.
"""

from __future__ import annotations
from typing import List
import numpy as np

from .waypoint_trajectory import Waypoint


# ─────────────────────── Trajectory 1: Square ─────────────────────────
def square_trajectory(side: float = 2.0, hold: float = 1.5) -> List[Waypoint]:
    """4 corners of a side x side square. Robot faces along each edge.

    Tests: 90° in-place turns, start/stop, straight-line tracking. The hold
    time gives the controller time to settle yaw between segments.
    """
    s = side
    return [
        Waypoint(x=s,   y=0.0, yaw=0.0,           hold_time=hold),
        Waypoint(x=s,   y=s,   yaw=np.pi / 2,     hold_time=hold),
        Waypoint(x=0.0, y=s,   yaw=np.pi,         hold_time=hold),
        Waypoint(x=0.0, y=0.0, yaw=-np.pi / 2,    hold_time=hold),
        Waypoint(x=0.0, y=0.0, yaw=0.0,           hold_time=0.0),
    ]


# ─────────────────────── Trajectory 2: Figure-8 ───────────────────────
def figure8_trajectory(radius: float = 1.0, n_per_loop: int = 8) -> List[Waypoint]:
    """Two lobes joined at the origin, sampled as waypoints along each circle.

    Tests: simultaneous (vx, wz) tracking, smooth continuous turning in
    alternating directions.
    """
    waypoints: List[Waypoint] = []

    # Right lobe (positive y), counter-clockwise.
    for k in range(1, n_per_loop + 1):
        theta = -np.pi / 2 + 2 * np.pi * k / n_per_loop
        cx, cy = 0.0, radius
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        yaw = _wrap_pi(theta + np.pi / 2)
        waypoints.append(Waypoint(x=x, y=y, yaw=yaw, hold_time=0.0))

    # Left lobe (negative y), clockwise.
    for k in range(1, n_per_loop + 1):
        theta = np.pi / 2 - 2 * np.pi * k / n_per_loop
        cx, cy = 0.0, -radius
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        yaw = _wrap_pi(theta - np.pi / 2)
        waypoints.append(Waypoint(x=x, y=y, yaw=yaw, hold_time=0.0))

    return waypoints


# ─────────────────────── Trajectory 3: Line with stops ────────────────
def line_with_stops_trajectory(distance: float = 2.5,
                                hold: float = 2.0) -> List[Waypoint]:
    """Out-and-back with pauses at each endpoint.

    Tests: pure forward/backward velocity tracking, stopping accuracy.
    Cleanest baseline — easy to read off the velocity tracking quality.
    """
    return [
        Waypoint(x=distance,        y=0.0, yaw=0.0, hold_time=hold),
        Waypoint(x=distance / 2,    y=0.0, yaw=0.0, hold_time=hold),
        Waypoint(x=distance,        y=0.0, yaw=0.0, hold_time=hold),
        Waypoint(x=0.0,             y=0.0, yaw=0.0, hold_time=hold),
    ]


# ─────────────────────── Registry ─────────────────────────────────────
TRAJECTORIES = {
    "square":   square_trajectory,
    "figure8":  figure8_trajectory,
    "line":     line_with_stops_trajectory,
}


def get_trajectory(name: str) -> List[Waypoint]:
    """Look up a trajectory by name. Accepts 'square', 'figure8', 'line'."""
    name = name.lower()
    if name not in TRAJECTORIES:
        raise KeyError(f"Unknown trajectory '{name}'. "
                       f"Available: {list(TRAJECTORIES.keys())}")
    return TRAJECTORIES[name]()


# ─────────────────────── helpers ──────────────────────────────────────
def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)
