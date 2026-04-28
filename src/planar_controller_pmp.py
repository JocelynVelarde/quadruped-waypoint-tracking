"""PMP controller for planar waypoint tracking.

Implements the Pontryagin Maximum Principle on the 6-dim planar body state.
Uses the steady-state Riccati solution (infinite-horizon) for real-time
feedback, same as the professor's controller but on the reduced planar model.

Outputs 3-dim velocity commands [vx_cmd, vy_cmd, wz_cmd].
"""

import numpy as np
from scipy.linalg import solve_discrete_are


class PlanarPMPController:
    """PMP-based controller for planar body waypoint tracking.

    Uses the steady-state costate solution:
        u* = -R⁻¹ Bᵀ P x  where P solves the DARE

    This is equivalent to LQR in steady-state, derived from PMP's
    optimality condition: u* = argmin H = -R⁻¹ Bᵀ λ

    Parameters
    ----------
    A, B  : discrete-time planar dynamics (6×6, 6×3)
    Q     : state cost (6×6)
    R     : control cost (3×3)
    dt    : control timestep
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray = None, R: np.ndarray = None,
                 Q_f: np.ndarray = None,
                 dt: float = 0.01,
                 v_max: float = 0.5, w_max: float = 0.6):
        self.A = A
        self.B = B
        self.dt = dt
        self.v_max = v_max
        self.w_max = w_max
        self.nx = A.shape[0]
        self.nu = B.shape[1]

        if Q is None:
            Q = np.diag([80, 80, 120, 5, 5, 10])
        if R is None:
            R = np.diag([0.5, 0.5, 0.3])
        if Q_f is None:
            Q_f = Q * 3.0

        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.R_inv = np.linalg.inv(R)

        # Solve DARE — PMP steady-state costate P satisfies the same equation
        # as LQR. The costate λ = P (x - x_ref), u* = -R⁻¹ Bᵀ λ
        self.P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ self.P @ B) @ B.T @ self.P @ A

        # Integrated velocity output
        self._vel_cmd = np.zeros(3)
        self._initialized = False

    def reset(self, x0: np.ndarray):
        self._vel_cmd = x0[3:6].copy()
        self._initialized = True

    def compute_control(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Compute PMP optimal control. Returns velocity command [vx, vy, wz].

        PMP optimality condition:
            u* = -R⁻¹ Bᵀ λ  where  λ = P (x - x_ref)

        Parameters
        ----------
        x     : current state (6,)
        x_ref : reference state (6,)

        Returns
        -------
        vel_cmd : np.ndarray (3,) — [vx_cmd, vy_cmd, wz_cmd]
        """
        if not self._initialized:
            self.reset(x)

        dx = x - x_ref
        # Wrap yaw error to [-pi, pi]
        dx[2] = float(np.arctan2(np.sin(dx[2]), np.cos(dx[2])))

        # PMP: costate λ = P dx, optimal control u* = -R⁻¹ Bᵀ λ
        lam = self.P @ dx
        u = -self.R_inv @ self.B.T @ lam

        # Integrate to velocity command
        self._vel_cmd = self._vel_cmd + u * self.dt

        # Saturate
        self._vel_cmd[0] = np.clip(self._vel_cmd[0], -self.v_max, self.v_max)
        self._vel_cmd[1] = np.clip(self._vel_cmd[1], -self.v_max, self.v_max)
        self._vel_cmd[2] = np.clip(self._vel_cmd[2], -self.w_max, self.w_max)

        return self._vel_cmd.copy()
