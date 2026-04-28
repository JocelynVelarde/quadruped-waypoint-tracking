"""State estimators for the decentralized framework (Kang et al. Sec IV).

1. OrientationEKF  – Orientation estimation fusing IMU gyro + accelerometer
                     (paper Eq. 26–30). Provides R̂_WB to the velocity MHE / controllers.

2. KalmanFilter    – Full discrete-time Kalman filter on the linearized SRB
                     dynamics for LQG control.  Implements the standard predict/update
                     cycle with process noise Q and measurement noise R.
"""

import numpy as np
from scipy.spatial.transform import Rotation


# ======================================================================
# 1.  Orientation EKF  (Paper Section IV.A, Eq. 26-30)
# ======================================================================
class OrientationEKF:
    """Quaternion-based EKF for floating-base orientation.

    Process model (Eq. 26):
        q⁺ = (I + ½ Ω(ω) Δt) q + W(q) δω

    Measurement model (Eq. 27-28):
        y_a = R_WB^T g + κ δ_a    (gravity in body frame)
    """

    def __init__(self, dt: float = 0.005,
                 gyro_noise: float = 0.01,
                 accel_noise: float = 0.05,
                 gyro_bias_noise: float = 0.001):
        self.dt = dt

        # State: [qw, qx, qy, qz, bω_x, bω_y, bω_z]  (7-dim)
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(7) * 0.01

        # Process noise
        self.Q = np.diag([
            gyro_noise, gyro_noise, gyro_noise, gyro_noise,  # quaternion
            gyro_bias_noise, gyro_bias_noise, gyro_bias_noise,  # bias
        ])

        # Measurement noise (accelerometer)
        self.R_accel = np.eye(3) * accel_noise ** 2

        self.g_world = np.array([0.0, 0.0, -9.81])

    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        """Quaternion (wxyz) → rotation matrix."""
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        return Rotation.from_quat(q_xyzw).as_matrix()

    def _omega_matrix(self, omega: np.ndarray) -> np.ndarray:
        """Ω(ω) matrix for quaternion kinematics (Eq. 26)."""
        wx, wy, wz = omega
        return np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0],
        ])

    def predict(self, gyro: np.ndarray):
        """Propagate orientation using gyroscope reading (Eq. 26).

        Parameters
        ----------
        gyro : (3,) measured angular velocity (body frame, rad/s)
        """
        q = self.x[:4]
        b_omega = self.x[4:7]
        omega = gyro - b_omega  # bias-corrected

        # Process model: q⁺ = (I + ½ Ω(ω) Δt) q
        Omega = self._omega_matrix(omega)
        F_q = np.eye(4) + 0.5 * Omega * self.dt

        q_new = F_q @ q
        q_new /= np.linalg.norm(q_new)  # normalise

        # Bias: random walk (Eq. 12)
        b_new = b_omega  # bias stays constant in predict

        self.x = np.concatenate([q_new, b_new])

        # Jacobian (7×7 block-diagonal approximation)
        F = np.eye(7)
        F[:4, :4] = F_q
        # ∂q⁺/∂b_ω ≈ −½ W(q) Δt  (simplified)
        F[:4, 4:7] = -0.5 * self.dt * np.array([
            [-q[1], -q[2], -q[3]],
            [ q[0], -q[3],  q[2]],
            [ q[3],  q[0], -q[1]],
            [-q[2],  q[1],  q[0]],
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update_accel(self, accel: np.ndarray):
        """Correct orientation using accelerometer (Eq. 27-28).

        Parameters
        ----------
        accel : (3,) measured acceleration in body frame
        """
        q = self.x[:4]
        R = self._quat_to_rot(q)

        # Expected measurement: gravity projected into body frame
        g_body_expected = R.T @ self.g_world

        # Actual measurement (normalised by ||a||/||g||  as in Eq. 28)
        a_norm = np.linalg.norm(accel)
        if a_norm < 1e-6:
            return
        kappa = a_norm / np.linalg.norm(self.g_world)

        # Innovation
        y = accel - g_body_expected

        # Measurement Jacobian (3×7): ∂(R^T g)/∂q  – numerical approximation
        H = np.zeros((3, 7))
        eps = 1e-5
        for i in range(4):
            q_p = q.copy()
            q_p[i] += eps
            q_p /= np.linalg.norm(q_p)
            R_p = self._quat_to_rot(q_p)
            H[:, i] = (R_p.T @ self.g_world - g_body_expected) / eps
        # H[:, 4:7] = 0  (accel doesn't depend on gyro bias directly)

        R_meas = self.R_accel * kappa ** 2

        # Kalman gain
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        dx = K @ y
        self.x = self.x + dx
        self.x[:4] /= np.linalg.norm(self.x[:4])
        self.P = (np.eye(7) - K @ H) @ self.P

    @property
    def orientation_quat(self) -> np.ndarray:
        """Return estimated quaternion (wxyz)."""
        return self.x[:4].copy()

    @property
    def orientation_matrix(self) -> np.ndarray:
        """Return estimated R_WB."""
        return self._quat_to_rot(self.x[:4])

    @property
    def euler_rpy(self) -> np.ndarray:
        """Return estimated roll, pitch, yaw."""
        q = self.x[:4]
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        return Rotation.from_quat(q_xyzw).as_euler('xyz')

    @property
    def gyro_bias(self) -> np.ndarray:
        return self.x[4:7].copy()


# ======================================================================
# 2.  Full-state Kalman Filter (for LQG)
# ======================================================================
class KalmanFilter:
    """Discrete-time Kalman filter on the linearised SRB dynamics.

    State: x = [p(3), v(3), θ(3), ω(3)]  (12-dim)

    Process:  x_{k+1} = A x_k + B u_k + g + w,   w ~ N(0, Q_proc)
    Measure:  y_k     = C x_k + v,                v ~ N(0, R_meas)

    The measurement matrix C selects observable states. Typically we measure:
      - position (from leg odometry / VO, paper Eq. 19-20)
      - velocity (from leg kinematics, paper Eq. 21-22)
      - orientation (from orientation EKF)
    So C ≈ I₁₂ with varying noise levels.
    """

    def __init__(self, nx: int = 12, ny: int = 12,
                 Q_proc: np.ndarray = None, R_meas: np.ndarray = None):
        self.nx = nx
        self.ny = ny

        self.x_hat = np.zeros(nx)
        self.P = np.eye(nx) * 0.1

        self.Q = Q_proc if Q_proc is not None else np.diag(
            [0.001] * 3 +   # position process noise
            [0.01] * 3 +    # velocity process noise
            [0.005] * 3 +   # orientation process noise
            [0.01] * 3      # angular velocity process noise
        )

        self.R = R_meas if R_meas is not None else np.diag(
            [0.005] * 3 +   # position measurement noise
            [0.02] * 3 +    # velocity measurement noise
            [0.01] * 3 +    # orientation measurement noise
            [0.05] * 3      # angular velocity measurement noise
        )

        self.C = np.eye(ny, nx)  # full-state measurement by default

    def predict(self, A: np.ndarray, B: np.ndarray, u: np.ndarray, g: np.ndarray = None):
        """Time update (predict).

        x̂⁻ = A x̂ + B u + g
        P⁻  = A P A^T + Q
        """
        aff = g if g is not None else np.zeros(self.nx)
        self.x_hat = A @ self.x_hat + B @ u + aff
        self.P = A @ self.P @ A.T + self.Q

    def update(self, y: np.ndarray):
        """Measurement update (correct).

        K = P⁻ Cᵀ (C P⁻ Cᵀ + R)⁻¹
        x̂ = x̂⁻ + K (y − C x̂⁻)
        P = (I − K C) P⁻
        """
        C = self.C
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)

        innovation = y - C @ self.x_hat
        self.x_hat = self.x_hat + K @ innovation
        self.P = (np.eye(self.nx) - K @ C) @ self.P

    @property
    def state_estimate(self) -> np.ndarray:
        return self.x_hat.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()
