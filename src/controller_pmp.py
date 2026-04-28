"""Pontryagin Maximum Principle (PMP) optimal controller.

Mathematical formulation (Murrieta-Cid, HJB & PMP slides):

    Hamiltonian (Eq. 13):
        H(x, u, λ, t) = L(x, u, t) + ⟨λ, ẋ⟩

    Adjoint / costate equation (Eq. 24-25):
        λ̇* = −∂H/∂x

    Optimality condition (Eq. 26):
        H(x*, u*, λ*, t) = min_u H(x*, u, λ*, t)

    State equation (Eq. 27):
        ẋ* = ∂H/∂λ = f(x*, u*)

For the quadratic-cost LTI case:
    L(x, u) = ½ xᵀ Q_s x + ½ uᵀ R_u u
    ẋ = A x + B u + g

    ∂H/∂u = R_u u + Bᵀ λ = 0   →   u* = −R_u⁻¹ Bᵀ λ
    λ̇ = −Q_s x − Aᵀ λ

This yields a Two-Point Boundary Value Problem (TPBVP):
    [ẋ]   [  A    −B R⁻¹ Bᵀ ] [x]   [g]
    [λ̇] = [ −Q_s    −Aᵀ     ] [λ] + [0]

    x(0) = x₀,   λ(T) = Q_f x(T)    (transversality)

We solve this via:
  (a) Shooting method for finite-horizon problems
  (b) Steady-state Riccati for infinite-horizon (feedback form: u = −K x)

Both are implemented below.
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm


class PontryaginController:
    """PMP-based optimal controller for the linearised quadruped SRB model.

    Parameters
    ----------
    A, B : continuous-time system matrices (12×12, 12×12)
    Q_s  : state cost matrix (12×12)
    R_u  : control cost matrix (12×12)
    Q_f  : terminal cost matrix (12×12)
    dt   : discretisation time-step
    horizon : number of discrete steps for finite-horizon BVP
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q_s: np.ndarray, R_u: np.ndarray,
                 Q_f: np.ndarray = None,
                 g_aff: np.ndarray = None,
                 dt: float = 0.005,
                 horizon: int = 200):
        self.A = A
        self.B = B
        self.Q_s = Q_s
        self.R_u = R_u
        self.Q_f = Q_f if Q_f is not None else Q_s.copy()
        self.dt = dt
        self.horizon = horizon
        self.g_aff = g_aff if g_aff is not None else np.zeros(A.shape[0])

        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.R_u_inv = np.linalg.inv(R_u)

        # Pre-compute steady-state gain for infinite-horizon fallback
        self._compute_steady_state_gain()

        # Storage for the solved costate trajectory
        self._lambda_traj = None
        self._x_traj = None
        self._u_traj = None
        self._gains = None
        self._P_seq = None
        self._p_seq = None
        self._t_traj = None

    # ------------------------------------------------------------------
    # Steady-state (infinite-horizon) solution via CARE
    # ------------------------------------------------------------------
    def _compute_steady_state_gain(self):
        """Solve continuous-time ARE:  Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0

        Then  K = R⁻¹ Bᵀ P  and  u* = −K (x − x_ref)
        """
        try:
            P = solve_continuous_are(self.A, self.B, self.Q_s, self.R_u)
            self.K_ss = self.R_u_inv @ self.B.T @ P
            self.P_ss = P
        except Exception:
            # Fallback: use discrete ARE
            Ad = np.eye(self.nx) + self.A * self.dt
            Bd = self.B * self.dt
            P = solve_discrete_are(Ad, Bd, self.Q_s * self.dt, self.R_u * self.dt)
            self.K_ss = np.linalg.inv(self.R_u * self.dt + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
            self.P_ss = P

    # ------------------------------------------------------------------
    # Finite-horizon TPBVP via collocation  (scipy solve_bvp)
    # ------------------------------------------------------------------
    def solve_bvp(self, x0: np.ndarray, x_ref: np.ndarray = None) -> bool:
        """Solve the two-point BVP from PMP.

        State+costate ODE (augmented 2n system):
            d/dt [x; λ] = M [x; λ] + [g; 0]

        BCs:  x(0) = x0,  λ(T) = Q_f (x(T) − x_ref)

        Returns True if converged.
        """
        if x_ref is None:
            x_ref = np.zeros(self.nx)
        n = self.nx
        T = self.horizon * self.dt

        # Hamiltonian system matrix
        BRinvBT = self.B @ self.R_u_inv @ self.B.T
        M = np.block([
            [self.A,    -BRinvBT],
            [-self.Q_s, -self.A.T],
        ])

        g_aug = np.concatenate([self.g_aff + self.A @ (-x_ref), self.Q_s @ x_ref])

        def ode(t, z):
            """RHS of the augmented ODE."""
            dz = M @ z + g_aug
            return dz

        def bc(za, zb):
            """Boundary conditions: x(0)=x0, λ(T)=Q_f(x(T)−x_ref)."""
            res_a = za[:n] - x0   # x(0) = x0
            res_b = zb[n:] - self.Q_f @ (zb[:n] - x_ref)  # transversality
            return np.concatenate([res_a, res_b])

        # Initial guess: linear interpolation
        t_grid = np.linspace(0, T, self.horizon + 1)
        z_init = np.zeros((2 * n, len(t_grid)))
        for i, t in enumerate(t_grid):
            alpha = t / T
            z_init[:n, i] = x0 * (1 - alpha) + x_ref * alpha
            z_init[n:, i] = self.P_ss @ (z_init[:n, i] - x_ref)

        try:
            sol = solve_bvp(ode, bc, t_grid, z_init, tol=1e-4, max_nodes=5000)
            if sol.success:
                self._x_traj = sol.y[:n, :].T  # (N+1, 12)
                self._lambda_traj = sol.y[n:, :].T
                self._t_traj = sol.x
                # Compute optimal control: u* = −R⁻¹ Bᵀ λ
                self._u_traj = np.array([
                    -self.R_u_inv @ self.B.T @ lam for lam in self._lambda_traj
                ])
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Discrete-time backward sweep (Riccati-like, more robust)
    # ------------------------------------------------------------------
    def solve_discrete_sweep(self, x0: np.ndarray, x_ref: np.ndarray = None):
        """Backward costate sweep then forward simulation.

        Discrete adjoint equation (backward):
            λ_k = Q_s (x_k − x_ref) + A_dᵀ λ_{k+1}
            u_k = −R_u⁻¹ B_dᵀ λ_{k+1}

        This is solved iteratively: guess λ → forward pass → backward pass → repeat.
        We use the Riccati recursion which is equivalent.

        P_N = Q_f
        K_k = (R + Bᵀ P_{k+1} B)⁻¹ Bᵀ P_{k+1} A
        P_k = Q + Aᵀ P_{k+1} A − Aᵀ P_{k+1} B K_k
        """
        if x_ref is None:
            x_ref = np.zeros(self.nx)

        Ad = np.eye(self.nx) + self.A * self.dt
        Bd = self.B * self.dt
        g_d = self.g_aff * self.dt
        Q = self.Q_s * self.dt
        R = self.R_u * self.dt
        N = self.horizon

        # Backward Riccati recursion
        P = [None] * (N + 1)
        K = [None] * N
        p = [None] * (N + 1)  # affine term for tracking

        P[N] = self.Q_f
        p[N] = -self.Q_f @ x_ref

        for k in range(N - 1, -1, -1):
            BtP = Bd.T @ P[k + 1]
            K[k] = np.linalg.inv(R + BtP @ Bd) @ BtP @ Ad
            P[k] = Q + Ad.T @ P[k + 1] @ (Ad - Bd @ K[k])
            p[k] = -Q @ x_ref + (Ad - Bd @ K[k]).T @ (p[k + 1] + P[k + 1] @ g_d)

        # Forward simulation
        x_traj = np.zeros((N + 1, self.nx))
        u_traj = np.zeros((N, self.nu))
        x_traj[0] = x0

        for k in range(N):
            # u_k = −K_k x_k − R⁻¹ Bᵀ (p_{k+1} + P_{k+1} g_d)
            ff = np.linalg.inv(R + Bd.T @ P[k + 1] @ Bd) @ Bd.T @ (p[k + 1] + P[k + 1] @ g_d)
            u_traj[k] = -K[k] @ x_traj[k] - ff
            x_traj[k + 1] = Ad @ x_traj[k] + Bd @ u_traj[k] + g_d

        self._x_traj = x_traj
        self._u_traj = u_traj
        self._gains = K
        self._P_seq = P
        self._p_seq = p
        return K, P, p

    # ------------------------------------------------------------------
    # Online control (feedback)
    # ------------------------------------------------------------------
    def compute_control(self, x: np.ndarray, x_ref: np.ndarray = None,
                        u_ref: np.ndarray = None,
                        step_idx: int = None) -> np.ndarray:
        """Compute u* given current state.

        Uses PMP optimality: u* = −R⁻¹ Bᵀ λ  (Eq. 26 applied).

        If a BVP/sweep solution exists and step_idx is provided, use the
        time-varying gain. Otherwise fall back to steady-state.

        Parameters
        ----------
        x     : current state (12,)
        x_ref : reference state (12,)
        u_ref : reference control / feedforward (12,)
        step_idx : index into the solved trajectory

        Returns
        -------
        u : optimal control (12,)
        """
        if x_ref is None:
            x_ref = np.zeros(self.nx)
        if u_ref is None:
            u_ref = np.zeros(self.nu)

        dx = x - x_ref

        # Time-varying gain from discrete sweep
        if self._gains is not None and step_idx is not None:
            k = min(step_idx, len(self._gains) - 1)
            u = -self._gains[k] @ dx + u_ref
        else:
            # Steady-state: u = −K_ss dx + u_ref
            u = -self.K_ss @ dx + u_ref

        return u

    @property
    def optimal_trajectory(self):
        """Return solved (x_traj, u_traj) if available."""
        return self._x_traj, self._u_traj
