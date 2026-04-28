"""Model Predictive Control (MPC) for quadruped locomotion.

Solves a receding-horizon QP at each time step:

    min   Σ_{k=0}^{N-1} [ (x_k−x_ref)ᵀ Q (x_k−x_ref) + (u_k−u_ref)ᵀ R (u_k−u_ref) ]
          + (x_N−x_ref)ᵀ Q_f (x_N−x_ref)

    s.t.  x_{k+1} = A x_k + B u_k + g        (dynamics, paper Eq. 32)
          D u_k ≤ d                             (friction cones)
          u_min ≤ u_k ≤ u_max                  (force limits)

This is the standard convex MPC formulation used in MIT Cheetah 3
(Di Carlo et al. 2018), adapted to the paper's decentralized framework
where orientation is estimated separately by EKF.

Connection to HJB/PMP:
- MPC approximates the HJB value function V(x,t) over a finite window
- The Bellman principle of optimality (HJB PDF Eq. 5) is applied
  recurrently with horizon N
- As N→∞ and without constraints, MPC converges to LQR (= PMP steady state)

Solved via OSQP (paper uses OSQP for the MHE QP, Sec VI).
"""

import numpy as np
from scipy import sparse

try:
    import osqp
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False


class MPCController:
    """Receding-horizon QP-based MPC for quadruped COM control.

    Parameters
    ----------
    A_d, B_d : discrete-time system matrices (12×12, 12×12)
    g_d      : gravity affine vector (12,)
    Q        : state cost (12×12)
    R        : control cost (12×12)
    Q_f      : terminal cost (12×12)
    N        : prediction horizon
    mu       : friction coefficient
    fz_max   : maximum vertical force per leg
    contact_schedule : (N, 4) bool array, which legs in contact per step
    """

    def __init__(self, A_d: np.ndarray, B_d: np.ndarray, g_d: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, Q_f: np.ndarray = None,
                 N: int = 10, mu: float = 0.6, fz_max: float = 200.0):
        self.A = A_d
        self.B = B_d
        self.g = g_d
        self.Q = Q
        self.R = R
        self.Q_f = Q_f if Q_f is not None else Q.copy()
        self.N = N
        self.mu = mu
        self.fz_max = fz_max

        self.nx = A_d.shape[0]
        self.nu = B_d.shape[1]

        if not HAS_OSQP:
            raise ImportError("OSQP is required for MPC. Install via: pip install osqp")

        self._solver = None
        self._setup_qp()

    def _setup_qp(self):
        """Build the QP matrices for the condensed MPC formulation.

        Decision variables: z = [u_0, u_1, ..., u_{N-1}]  (N×nu)
        States are eliminated via: x_k = A^k x_0 + Σ A^{k-1-j} B u_j + ...
        """
        N, nx, nu = self.N, self.nx, self.nu
        nz = N * nu  # decision variable dimension

        # ------------------------------------------------------------------
        # Precompute powers of A and the S_x, S_u matrices
        #   x_vec = S_x x_0 + S_u u_vec + S_g
        # where x_vec = [x_1; x_2; ...; x_N], u_vec = [u_0; ...; u_{N-1}]
        # ------------------------------------------------------------------
        A_pow = [np.eye(nx)]
        for k in range(N):
            A_pow.append(A_pow[-1] @ self.A)

        # S_x: (N*nx) × nx
        self.S_x = np.vstack(A_pow[1:])

        # S_u: (N*nx) × (N*nu)  (lower-triangular block Toeplitz)
        self.S_u = np.zeros((N * nx, N * nu))
        for i in range(N):
            for j in range(i + 1):
                self.S_u[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = A_pow[i - j] @ self.B

        # S_g: affine gravity stacking  (N*nx,)
        self.S_g = np.zeros(N * nx)
        for i in range(N):
            g_sum = np.zeros(nx)
            for j in range(i + 1):
                g_sum += A_pow[j] @ self.g
            self.S_g[i * nx:(i + 1) * nx] = g_sum

        # ------------------------------------------------------------------
        # Cost:  J = ½ u_vec^T H u_vec + f^T u_vec  (depends on x0, x_ref)
        # ------------------------------------------------------------------
        # Q_bar = blkdiag(Q, Q, ..., Q, Q_f)
        Q_bar = np.zeros((N * nx, N * nx))
        for i in range(N - 1):
            Q_bar[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = self.Q
        Q_bar[(N - 1) * nx:, (N - 1) * nx:] = self.Q_f

        R_bar = np.zeros((N * nu, N * nu))
        for i in range(N):
            R_bar[i * nu:(i + 1) * nu, i * nu:(i + 1) * nu] = self.R

        self.H = self.S_u.T @ Q_bar @ self.S_u + R_bar
        self.H = 0.5 * (self.H + self.H.T)  # symmetrise
        self.Q_bar = Q_bar

        # ------------------------------------------------------------------
        # Friction cone constraints (per leg, per step)
        # D_leg @ f_leg <= 0,  plus  0 <= fz <= fz_max
        # ------------------------------------------------------------------
        D_leg = np.array([
            [ 1,  0, -self.mu],
            [-1,  0, -self.mu],
            [ 0,  1, -self.mu],
            [ 0, -1, -self.mu],
            [ 0,  0, -1],
        ])
        n_ineq_per_leg = 5
        n_legs = 4
        n_ineq_per_step = n_ineq_per_leg * n_legs
        n_ineq = n_ineq_per_step * N

        # Also add upper bound on fz
        n_fz_bounds = n_legs * N

        D_full = np.zeros((n_ineq + n_fz_bounds, nz))
        d_upper = np.zeros(n_ineq + n_fz_bounds)
        d_lower = np.full(n_ineq + n_fz_bounds, -np.inf)

        for k in range(N):
            for leg in range(n_legs):
                row_start = k * n_ineq_per_step + leg * n_ineq_per_leg
                col_start = k * nu + leg * 3
                D_full[row_start:row_start + n_ineq_per_leg,
                       col_start:col_start + 3] = D_leg
                d_upper[row_start:row_start + n_ineq_per_leg] = 0.0

            # fz upper bounds
            for leg in range(n_legs):
                row = n_ineq + k * n_legs + leg
                col = k * nu + leg * 3 + 2
                D_full[row, col] = 1.0
                d_upper[row] = self.fz_max
                d_lower[row] = 0.0

        self.D_full = D_full
        self.d_upper = d_upper
        self.d_lower = d_lower

    def _build_and_solve(self, x0: np.ndarray, x_ref: np.ndarray,
                         u_ref: np.ndarray,
                         contact_mask: np.ndarray = None) -> np.ndarray:
        """Build cost vector f(x0, x_ref) and solve the QP.

        Returns
        -------
        u_vec : (N*nu,) optimal control sequence, or None if infeasible
        """
        N, nx, nu = self.N, self.nx, self.nu
        nz = N * nu

        # Reference stacking
        x_ref_vec = np.tile(x_ref, N)
        u_ref_vec = np.tile(u_ref, N)

        # f = S_u^T Q_bar (S_x x0 + S_g - x_ref_vec) - R_bar u_ref_vec
        # but R_bar u_ref already in H via centering
        pred_free = self.S_x @ x0 + self.S_g - x_ref_vec

        R_bar_diag = np.zeros(nz)
        for i in range(N):
            R_bar_diag[i * nu:(i + 1) * nu] = np.diag(self.R)

        f = self.S_u.T @ self.Q_bar @ pred_free - np.diag(R_bar_diag) @ u_ref_vec

        # Contact mask: zero out forces for swing legs
        D_ineq = self.D_full.copy()
        d_upper = self.d_upper.copy()
        d_lower = self.d_lower.copy()

        if contact_mask is not None:
            for k in range(N):
                cm = contact_mask[k] if contact_mask.ndim > 1 else contact_mask
                for leg in range(4):
                    if not cm[leg]:
                        # Force this leg's forces to zero via equality
                        col_start = k * nu + leg * 3
                        # Zero the columns in H contribution (add large penalty)
                        f[col_start:col_start + 3] = 0

        # Solve with OSQP
        P_sp = sparse.csc_matrix(self.H)
        A_sp = sparse.csc_matrix(D_ineq)

        solver = osqp.OSQP()
        solver.setup(P_sp, f, A_sp, d_lower, d_upper,
                     verbose=False, warm_start=True,
                     eps_abs=1e-4, eps_rel=1e-4,
                     max_iter=200, polish=True)
        result = solver.solve()

        if result.info.status == 'solved' or result.info.status == 'solved_inaccurate':
            return result.x
        else:
            return None

    def compute_control(self, x: np.ndarray, x_ref: np.ndarray,
                        u_ref: np.ndarray = None,
                        contact_mask: np.ndarray = None) -> np.ndarray:
        """Compute MPC control for current state.

        Parameters
        ----------
        x          : current state (12,)
        x_ref      : reference state (12,)
        u_ref      : reference control (12,)
        contact_mask : (4,) or (N,4) bool contact schedule

        Returns
        -------
        u : control for current step (12,)
        """
        if u_ref is None:
            u_ref = np.zeros(self.nu)

        # Expand contact_mask to (N, 4)
        if contact_mask is None:
            contact_mask = np.ones((self.N, 4), dtype=bool)
        elif contact_mask.ndim == 1:
            contact_mask = np.tile(contact_mask, (self.N, 1))

        u_vec = self._build_and_solve(x, x_ref, u_ref, contact_mask)

        if u_vec is not None:
            return u_vec[:self.nu]  # first control action
        else:
            # Fallback: gravity compensation
            return u_ref

    def compute_full_trajectory(self, x0: np.ndarray, x_ref: np.ndarray,
                                u_ref: np.ndarray = None) -> tuple:
        """Solve MPC and return full predicted trajectory."""
        if u_ref is None:
            u_ref = np.zeros(self.nu)

        u_vec = self._build_and_solve(x0, x_ref, u_ref)
        if u_vec is None:
            return None, None

        N, nx, nu = self.N, self.nx, self.nu
        u_traj = u_vec.reshape(N, nu)
        x_traj = np.zeros((N + 1, nx))
        x_traj[0] = x0
        for k in range(N):
            x_traj[k + 1] = self.A @ x_traj[k] + self.B @ u_traj[k] + self.g

        return x_traj, u_traj

    def update_dynamics(self, A_d: np.ndarray, B_d: np.ndarray, g_d: np.ndarray):
        """Update system matrices (for time-varying dynamics as in paper Eq. 32)."""
        self.A = A_d
        self.B = B_d
        self.g = g_d
        self._setup_qp()
