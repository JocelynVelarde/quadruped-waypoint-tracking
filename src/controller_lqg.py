"""LQG (Linear Quadratic Gaussian) controller.

Combines:
  - LQR  : optimal state-feedback gain  u = −K(x̂ − x_ref) + u_ref
  - Kalman Filter : optimal state estimator  x̂

This is the separation principle: the LQR gain and Kalman gain can be
designed independently and combined.

The LQR solves the infinite-horizon cost:
    J = Σ (x−x_ref)ᵀ Q (x−x_ref) + (u−u_ref)ᵀ R (u−u_ref)

via the Discrete Algebraic Riccati Equation (DARE):
    P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A
    K = (R + Bᵀ P B)⁻¹ Bᵀ P A

The Kalman filter provides x̂ (see estimator_ekf.py KalmanFilter).

Connection to PMP: The LQR gain K is identical to the steady-state
costate solution λ = P x, u* = −R⁻¹ Bᵀ P x  (infinite-horizon PMP).
"""

import numpy as np
from scipy.linalg import solve_discrete_are

from .estimator_ekf import KalmanFilter


class LQGController:
    """LQG = LQR + Kalman Filter for the linearised quadruped model.

    Parameters
    ----------
    A_d, B_d : discrete-time system matrices
    g_d      : gravity affine vector
    Q        : state cost (12×12)
    R        : control cost (12×12)
    Q_proc   : process noise covariance for Kalman filter
    R_meas   : measurement noise covariance for Kalman filter
    """

    def __init__(self, A_d: np.ndarray, B_d: np.ndarray,
                 g_d: np.ndarray,
                 Q: np.ndarray, R: np.ndarray,
                 Q_proc: np.ndarray = None,
                 R_meas: np.ndarray = None):
        self.A_d = A_d
        self.B_d = B_d
        self.g_d = g_d
        self.Q = Q
        self.R = R
        self.nx = A_d.shape[0]
        self.nu = B_d.shape[1]

        # Solve DARE for LQR gain
        self.P = solve_discrete_are(A_d, B_d, Q, R)
        self.K = np.linalg.inv(R + B_d.T @ self.P @ B_d) @ B_d.T @ self.P @ A_d

        # Initialize Kalman filter
        self.kf = KalmanFilter(
            nx=self.nx, ny=self.nx,
            Q_proc=Q_proc, R_meas=R_meas,
        )

        # Feedforward for gravity compensation
        # At steady state: x_ref = A x_ref + B u_ff + g  →  u_ff = B⁺ (I−A) x_ref − B⁺ g
        # For standing: simplified to distribute weight equally
        self._u_ff = None

    def set_initial_estimate(self, x0: np.ndarray):
        """Set the Kalman filter initial state."""
        self.kf.x_hat = x0.copy()

    def compute_feedforward(self, x_ref: np.ndarray) -> np.ndarray:
        """Compute gravity-compensating feedforward at reference."""
        # (I - A) x_ref - g = B u_ff  →  u_ff = B⁺ ((I-A) x_ref - g)
        rhs = (np.eye(self.nx) - self.A_d) @ x_ref - self.g_d
        u_ff, _, _, _ = np.linalg.lstsq(self.B_d, rhs, rcond=None)
        return u_ff

    def step(self, y: np.ndarray, x_ref: np.ndarray,
             u_ref: np.ndarray = None) -> np.ndarray:
        """One step of LQG: update estimate, compute control.

        Parameters
        ----------
        y     : measurement vector (12,)
        x_ref : reference state (12,)
        u_ref : feedforward control (12,), if None computed automatically

        Returns
        -------
        u : control action (12,)
        """
        # 1. Kalman measurement update
        self.kf.update(y)

        # 2. LQR control on estimated state
        x_hat = self.kf.state_estimate
        dx = x_hat - x_ref

        if u_ref is None:
            u_ref = self.compute_feedforward(x_ref)

        u = -self.K @ dx + u_ref

        # 3. Kalman prediction for next step
        self.kf.predict(self.A_d, self.B_d, u, self.g_d)

        return u

    def compute_control(self, x: np.ndarray, x_ref: np.ndarray,
                        u_ref: np.ndarray = None) -> np.ndarray:
        """Direct state-feedback LQR (no Kalman, for when true state is available)."""
        dx = x - x_ref
        if u_ref is None:
            u_ref = self.compute_feedforward(x_ref)
        return -self.K @ dx + u_ref

    @property
    def state_estimate(self) -> np.ndarray:
        return self.kf.state_estimate

    @property
    def lqr_gain(self) -> np.ndarray:
        return self.K.copy()

    @property
    def riccati_solution(self) -> np.ndarray:
        return self.P.copy()
