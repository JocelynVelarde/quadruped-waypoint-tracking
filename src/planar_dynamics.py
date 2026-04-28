"""Planar body dynamics for waypoint-tracking controllers.

Models the robot body as a planar rigid body with state:
    x = [px, py, yaw, vx, vy, wz]   (6-dim)

Control input is commanded acceleration:
    u = [ax, ay, alpha]               (3-dim)

This is the "inverted pendulum" abstraction the professor described:
the controller decides how to accelerate the body toward the waypoint,
and the walking stack (Quadruped-PyMPC) realizes that velocity command.

Discrete-time model:
    x[k+1] = A x[k] + B u[k]

where A and B are computed from dt using ZOH integration.
"""

import numpy as np


class PlanarBodyDynamics:
    """6-state planar body dynamics for waypoint-tracking controllers.

    State:   x = [px, py, yaw, vx, vy, wz]
    Control: u = [ax, ay, alpha]  (body-frame accelerations)
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.nx = 6
        self.nu = 3
        self._build_matrices()

    def _build_matrices(self):
        dt = self.dt
        # Discrete-time A: integrates velocity into position
        self.A = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ],
        ])
        # Discrete-time B: acceleration → velocity change
        self.B = np.array([
            [0,  0,  0 ],
            [0,  0,  0 ],
            [0,  0,  0 ],
            [dt, 0,  0 ],
            [0,  dt, 0 ],
            [0,  0,  dt],
        ])
        self.g = np.zeros(self.nx)  # no gravity in planar model

    def get_reference(self, wp_x: float, wp_y: float, wp_yaw: float,
                      cmd_vx: float, cmd_vy: float, cmd_wz: float) -> np.ndarray:
        """Build the 6-dim reference state from a waypoint."""
        return np.array([wp_x, wp_y, wp_yaw, cmd_vx, cmd_vy, cmd_wz])

    def state_from_env(self, base_pos: np.ndarray,
                       base_ori_euler_xyz: np.ndarray,
                       base_lin_vel: np.ndarray,
                       base_ang_vel: np.ndarray) -> np.ndarray:
        """Extract 6-dim planar state from Quadruped-PyMPC env readings."""
        return np.array([
            float(base_pos[0]),
            float(base_pos[1]),
            float(base_ori_euler_xyz[2]),   # yaw
            float(base_lin_vel[0]),
            float(base_lin_vel[1]),
            float(base_ang_vel[2]),          # yaw rate
        ])
