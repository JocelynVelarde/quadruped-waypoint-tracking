"""Linearized Single Rigid Body (SRB) dynamics for quadruped COM control.

Follows the decentralized framework of Kang et al. (arXiv:2405.20567):
- Orientation R_WB is estimated separately (EKF, Sec IV.A)
- Position/velocity evolves on TIME-VARYING LINEAR dynamics (Eq. 32)

State vector (13 dims, using paper Eq. 31 convention adapted for COM):
    x = [p(3), v(3), θ_rpy(3), ω(3), b_a(1)]
    where p = COM position, v = COM velocity, θ = roll/pitch/yaw,
    ω = body angular velocity, b_a = scalar accel bias proxy.

    For the controllers we use the reduced 12-state version (no bias):
    x = [p(3), v(3), θ(3), ω(3)]

Control vector (12 dims):
    u = [f_FL(3), f_FR(3), f_RL(3), f_RR(3)]  -- ground reaction forces

Continuous dynamics:
    ṗ  = v
    v̇  = (1/m) Σ f_i + g                        (Newton)
    θ̇  = T(θ) ω                                  (Euler-angle rates)
    ω̇  = I⁻¹ (Σ r_i × f_i − ω × I ω)           (Euler's equation)

Linearised about standing pose (θ≈0 → T≈I₃):
    x_{k+1} = A_d x_k + B_d u_k + g_d
"""

import numpy as np
from scipy.spatial.transform import Rotation


class QuadrupedDynamics:
    """Linearized SRB dynamics, consistent with paper Eq. 32."""

    def __init__(self, mass: float = 12.0, inertia: np.ndarray = None, dt: float = 0.005):
        self.mass = mass
        self.dt = dt
        self.g = np.array([0.0, 0.0, -9.81])
        self.nx = 12  # state dimension
        self.nu = 12  # control dimension (4 legs × 3 forces)

        if inertia is None:
            # Typical Go1 / Aliengo inertia (kg·m²)
            self.I_body = np.diag([0.07, 0.26, 0.242])
        else:
            self.I_body = np.array(inertia)

        self.I_body_inv = np.linalg.inv(self.I_body)

        # Default foot positions relative to COM in body frame (standing pose)
        self.r_feet_body = np.array([
            [ 0.183,  0.132, -0.30],  # FL
            [ 0.183, -0.132, -0.30],  # FR
            [-0.183,  0.132, -0.30],  # RL
            [-0.183, -0.132, -0.30],  # RR
        ])

    # ------------------------------------------------------------------
    # Reference / equilibrium
    # ------------------------------------------------------------------
    def standing_state(self, height: float = 0.30) -> np.ndarray:
        """Nominal standing state."""
        x_ref = np.zeros(self.nx)
        x_ref[2] = height
        return x_ref

    def standing_control(self) -> np.ndarray:
        """Gravity-compensating GRFs (each leg supports mass/4)."""
        fz = -self.mass * self.g[2] / 4.0
        u_ref = np.zeros(self.nu)
        for i in range(4):
            u_ref[3 * i + 2] = fz
        return u_ref

    # ------------------------------------------------------------------
    # Euler-angle rate matrix  θ̇ = T(θ) ω
    # ------------------------------------------------------------------
    @staticmethod
    def euler_rate_matrix(rpy: np.ndarray) -> np.ndarray:
        """Maps body angular velocity to Euler-angle rates (XYZ convention)."""
        r, p, _ = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp = np.cos(p)
        if abs(cp) < 1e-8:
            cp = 1e-8
        T = np.array([
            [1, sr * np.tan(p), cr * np.tan(p)],
            [0, cr,             -sr],
            [0, sr / cp,        cr / cp],
        ])
        return T

    # ------------------------------------------------------------------
    # Rotation from Euler angles (world ← body)
    # ------------------------------------------------------------------
    @staticmethod
    def rotation_matrix(rpy: np.ndarray) -> np.ndarray:
        return Rotation.from_euler('xyz', rpy).as_matrix()

    # ------------------------------------------------------------------
    # Continuous-time A, B at a given linearization point
    # Following paper Eq. 32: the orientation R̂_WB is treated as known
    # ------------------------------------------------------------------
    def continuous_AB(self, x: np.ndarray, contact_mask: np.ndarray = None,
                      r_feet_world: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Build continuous-time Jacobians (A_c, B_c).

        Parameters
        ----------
        x : (12,) state
        contact_mask : (4,) bool – which legs are in contact
        r_feet_world : (4,3) foot positions in world frame (if None, computed from x)

        Returns
        -------
        A_c : (12,12)
        B_c : (12,12)
        """
        if contact_mask is None:
            contact_mask = np.ones(4, dtype=bool)

        rpy = x[6:9]
        R_WB = self.rotation_matrix(rpy)

        # Foot positions relative to COM in world frame
        if r_feet_world is None:
            p_com = x[0:3]
            r_feet_world = p_com + (R_WB @ self.r_feet_body.T).T

        r_rel = r_feet_world - x[0:3]  # (4,3) vectors from COM to feet

        A_c = np.zeros((12, 12))
        B_c = np.zeros((12, 12))

        # ṗ = v
        A_c[0:3, 3:6] = np.eye(3)

        # θ̇ = T(θ) ω  →  linearised: ∂/∂ω = T(θ), ∂/∂θ ≈ 0 at small angles
        T = self.euler_rate_matrix(rpy)
        A_c[6:9, 9:12] = T

        # v̇ = (1/m) Σ f_i + g   → B rows
        for i in range(4):
            if contact_mask[i]:
                B_c[3:6, 3 * i: 3 * i + 3] = np.eye(3) / self.mass

        # ω̇ = I⁻¹ (Σ r_i × f_i)  → B rows (linearised, dropping ω×Iω)
        for i in range(4):
            if contact_mask[i]:
                ri = r_rel[i]
                # Skew-symmetric matrix [r]×
                skew = np.array([
                    [0, -ri[2], ri[1]],
                    [ri[2], 0, -ri[0]],
                    [-ri[1], ri[0], 0],
                ])
                B_c[9:12, 3 * i: 3 * i + 3] = self.I_body_inv @ skew

        return A_c, B_c

    # ------------------------------------------------------------------
    # Gravity affine term
    # ------------------------------------------------------------------
    def gravity_vector(self) -> np.ndarray:
        """Affine gravity contribution g_d in x_{k+1} = A x + B u + g_d."""
        g_c = np.zeros(12)
        g_c[3:6] = self.g
        return g_c * self.dt

    # ------------------------------------------------------------------
    # Discretise  (ZOH)  x_{k+1} = A_d x_k + B_d u_k + g_d
    # ------------------------------------------------------------------
    def discretize(self, A_c: np.ndarray, B_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Zero-order hold discretisation (first-order approx for speed)."""
        A_d = np.eye(self.nx) + A_c * self.dt
        B_d = B_c * self.dt
        return A_d, B_d

    def get_linear_system(self, x: np.ndarray = None,
                          contact_mask: np.ndarray = None,
                          r_feet_world: np.ndarray = None):
        """Return (A_d, B_d, g_d) ready for controllers."""
        if x is None:
            x = self.standing_state()
        A_c, B_c = self.continuous_AB(x, contact_mask, r_feet_world)
        A_d, B_d = self.discretize(A_c, B_c)
        g_d = self.gravity_vector()
        return A_d, B_d, g_d

    # ------------------------------------------------------------------
    # Simulate one step (for testing without MuJoCo)
    # ------------------------------------------------------------------
    def step(self, x: np.ndarray, u: np.ndarray,
             contact_mask: np.ndarray = None) -> np.ndarray:
        """Forward-simulate one discrete step."""
        A_d, B_d, g_d = self.get_linear_system(x, contact_mask)
        return A_d @ x + B_d @ u + g_d

    # ------------------------------------------------------------------
    # Friction cone constraint matrices  (for MPC)
    #   |f_x| <= μ f_z,  |f_y| <= μ f_z,  f_z >= 0
    # Written as  D_i f_i <= 0  per leg
    # ------------------------------------------------------------------
    @staticmethod
    def friction_cone_constraints(mu: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (D, d) such that D @ u <= d encodes the friction pyramid
        for all 4 legs (20 rows × 12 cols).
        """
        # Per-leg: 5 inequalities
        D_leg = np.array([
            [ 1,  0, -mu],   # fx <= mu fz
            [-1,  0, -mu],   # -fx <= mu fz
            [ 0,  1, -mu],   # fy <= mu fz
            [ 0, -1, -mu],   # -fy <= mu fz
            [ 0,  0, -1],    # fz >= 0  →  -fz <= 0
        ])
        d_leg = np.zeros(5)

        D = np.zeros((20, 12))
        d = np.zeros(20)
        for i in range(4):
            D[5 * i: 5 * i + 5, 3 * i: 3 * i + 3] = D_leg
            d[5 * i: 5 * i + 5] = d_leg
        return D, d
