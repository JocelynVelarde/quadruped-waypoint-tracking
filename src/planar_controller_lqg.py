"""LQG controller for planar waypoint tracking.

Operates on the 6-dim planar body state [px, py, yaw, vx, vy, wz].
Outputs 3-dim velocity commands [vx_cmd, vy_cmd, wz_cmd] by integrating
the computed acceleration command.

This is the same LQG formulation as the professor's controller, applied
to the reduced planar model instead of the full 12-GRF formulation.
"""

import numpy as np
from scipy.linalg import solve_discrete_are


class PlanarLQGController:
    """LQG (LQR + Kalman Filter) for planar body waypoint tracking.

    Parameters
    ----------
    A, B : discrete-time planar dynamics (6×6, 6×3)
    Q    : state cost (6×6) — penalizes position and heading error
    R    : control cost (3×3) — penalizes aggressive acceleration
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray = None, R: np.ndarray = None,
                 dt: float = 0.01,
                 v_max: float = 0.5, w_max: float = 0.6):
        self.A = A
        self.B = B
        self.dt = dt
        self.v_max = v_max
        self.w_max = w_max
        self.nx = A.shape[0]
        self.nu = B.shape[1]

        # Default cost weights — tune position tracking more than velocity
        if Q is None:
            Q = np.diag([
                80, 80, 120,    # position (px, py, yaw) — high weight
                5,  5,  10,     # velocity (vx, vy, wz) — lower weight
            ])
        if R is None:
            R = np.diag([0.5, 0.5, 0.3])   # acceleration cost

        self.Q = Q
        self.R = R

        # Solve DARE for LQR gain
        self.P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ self.P @ B) @ B.T @ self.P @ A

        # Kalman filter covariances
        Q_proc = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])
        R_meas = np.diag([1e-3, 1e-3, 5e-3, 5e-3, 5e-3, 1e-2])
        self.Q_proc = Q_proc
        self.R_meas = R_meas

        # Kalman gain (steady-state)
        P_kf = solve_discrete_are(A.T, np.eye(self.nx), Q_proc, R_meas)
        self.L = P_kf @ np.linalg.inv(R_meas + P_kf)

        # State estimate
        self.x_hat = np.zeros(self.nx)
        self._initialized = False

        # Integrated velocity command (output)
        self._vel_cmd = np.zeros(3)

    def reset(self, x0: np.ndarray):
        self.x_hat = x0.copy()
        self._vel_cmd = x0[3:6].copy()
        self._initialized = True

    def step(self, x_measured: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """One LQG step. Returns velocity commands [vx, vy, wz].

        Parameters
        ----------
        x_measured : current measured state (6,)
        x_ref      : reference state (6,)

        Returns
        -------
        vel_cmd : np.ndarray (3,) — [vx_cmd, vy_cmd, wz_cmd]
        """
        if not self._initialized:
            self.reset(x_measured)

        # Kalman update
        innovation = x_measured - self.x_hat
        self.x_hat = self.x_hat + self.L @ innovation

        # LQR on estimated state
        dx = self.x_hat - x_ref
        # Wrap yaw error to [-pi, pi]
        dx[2] = float(np.arctan2(np.sin(dx[2]), np.cos(dx[2])))

        # Acceleration command
        u = -self.K @ dx

        # Integrate to velocity command
        self._vel_cmd = self._vel_cmd + u * self.dt

        # Kalman predict
        self.x_hat = self.A @ self.x_hat + self.B @ u

        # Saturate
        self._vel_cmd[0] = np.clip(self._vel_cmd[0], -self.v_max, self.v_max)
        self._vel_cmd[1] = np.clip(self._vel_cmd[1], -self.v_max, self.v_max)
        self._vel_cmd[2] = np.clip(self._vel_cmd[2], -self.w_max, self.w_max)

        return self._vel_cmd.copy()
