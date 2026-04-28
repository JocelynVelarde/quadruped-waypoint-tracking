"""MPC controller for planar waypoint tracking.

Receding-horizon QP on the 6-dim planar body state.
Outputs 3-dim velocity commands [vx_cmd, vy_cmd, wz_cmd].

Constraints:
  - Velocity saturation: |vx|, |vy| <= v_max, |wz| <= w_max
  - Acceleration saturation: |ax|, |ay| <= a_max, |alpha| <= alpha_max

Solved via OSQP.
"""

import numpy as np
from scipy import sparse

try:
    import osqp
    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False


class PlanarMPCController:
    """MPC for planar body waypoint tracking.

    Parameters
    ----------
    A, B  : discrete-time planar dynamics (6×6, 6×3)
    Q     : state cost (6×6)
    R     : control cost (3×3)
    Q_f   : terminal cost (6×6)
    N     : prediction horizon
    v_max : max linear velocity (m/s)
    w_max : max yaw rate (rad/s)
    a_max : max linear acceleration (m/s²)
    """

    def __init__(self, A: np.ndarray, B: np.ndarray,
                 Q: np.ndarray = None, R: np.ndarray = None,
                 Q_f: np.ndarray = None,
                 N: int = 15,
                 v_max: float = 0.5, w_max: float = 0.6,
                 a_max: float = 1.0, alpha_max: float = 1.5,
                 dt: float = 0.01):
        if not HAS_OSQP:
            raise ImportError("OSQP required: pip install osqp")

        self.A = A
        self.B = B
        self.N = N
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

        # Acceleration limits
        self.u_min = np.array([-a_max, -a_max, -alpha_max])
        self.u_max = np.array([ a_max,  a_max,  alpha_max])

        # Velocity limits (applied to state indices 3,4,5)
        self.v_min = np.array([-v_max, -v_max, -w_max])
        self.v_max_arr = np.array([ v_max,  v_max,  w_max])

        # Precompute rollout matrices
        self._build_rollout()

        # Integrated velocity output
        self._vel_cmd = np.zeros(3)

    def _build_rollout(self):
        N, nx, nu = self.N, self.nx, self.nu
        A_pow = [np.eye(nx)]
        for _ in range(N):
            A_pow.append(A_pow[-1] @ self.A)

        # S_x: (N*nx, nx)
        self.S_x = np.vstack(A_pow[1:])
        # S_u: (N*nx, N*nu) lower-triangular block Toeplitz
        self.S_u = np.zeros((N * nx, N * nu))
        for i in range(N):
            for j in range(i + 1):
                self.S_u[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_pow[i-j] @ self.B

        # Cost matrices
        Q_bar = np.zeros((N*nx, N*nx))
        for i in range(N - 1):
            Q_bar[i*nx:(i+1)*nx, i*nx:(i+1)*nx] = self.Q
        Q_bar[(N-1)*nx:, (N-1)*nx:] = self.Q_f
        R_bar = np.kron(np.eye(N), self.R)

        self.H = self.S_u.T @ Q_bar @ self.S_u + R_bar
        self.H = 0.5 * (self.H + self.H.T)
        self.Q_bar = Q_bar
        self.S_x_mat = self.S_x

    def compute_control(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        """Compute MPC control. Returns velocity command [vx, vy, wz]."""
        N, nx, nu = self.N, self.nx, self.nu
        nz = N * nu

        # Wrap yaw in state and reference
        x = x.copy()
        x_ref = x_ref.copy()

        x_ref_vec = np.tile(x_ref, N)
        # Wrap yaw errors
        for i in range(N):
            x_ref_vec[i*nx + 2] = x[2] + float(
                np.arctan2(np.sin(x_ref[2] - x[2]), np.cos(x_ref[2] - x[2])))

        # Gradient
        free_response = self.S_x @ x - x_ref_vec
        f = self.S_u.T @ self.Q_bar @ free_response

        # Box constraints on u: acceleration limits
        u_min_vec = np.tile(self.u_min, N)
        u_max_vec = np.tile(self.u_max, N)

        # Also constrain resulting velocities (state indices 3,4,5)
        # x_k = S_x[:,k] x0 + S_u[:k,:] u → velocity rows
        # For simplicity, use soft constraint via large R instead
        # Hard constraint: extract velocity rows from rollout
        vel_rows = []
        for i in range(N):
            for vi in range(3):
                row = np.zeros((1, nz))
                row[0, :] = self.S_u[(i*nx + 3 + vi), :]
                vel_rows.append(row)
        A_vel = np.vstack(vel_rows)
        # Free-response velocity
        x_free = self.S_x @ x
        b_vel_free = np.array([x_free[i*nx + 3 + vi]
                                for i in range(N) for vi in range(3)])

        v_lb = np.tile(self.v_min, N) - b_vel_free
        v_ub = np.tile(self.v_max_arr, N) - b_vel_free

        # Combine constraints
        A_con = np.vstack([np.eye(nz), A_vel])
        l_con = np.concatenate([u_min_vec, v_lb])
        u_con = np.concatenate([u_max_vec, v_ub])

        # Solve
        P_sp = sparse.csc_matrix(self.H)
        A_sp = sparse.csc_matrix(A_con)

        solver = osqp.OSQP()
        solver.setup(P_sp, f, A_sp, l_con, u_con,
                     verbose=False, warm_starting=True,
                     eps_abs=1e-4, eps_rel=1e-4,
                     max_iter=500, polish=True)
        result = solver.solve()

        if result.info.status in ('solved', 'solved_inaccurate'):
            u_opt = result.x[:nu]
        else:
            u_opt = np.zeros(nu)

        # Integrate acceleration → velocity
        self._vel_cmd = self._vel_cmd + u_opt * self.dt
        self._vel_cmd[0] = np.clip(self._vel_cmd[0], -self.v_max, self.v_max)
        self._vel_cmd[1] = np.clip(self._vel_cmd[1], -self.v_max, self.v_max)
        self._vel_cmd[2] = np.clip(self._vel_cmd[2], -self.w_max, self.w_max)

        return self._vel_cmd.copy()

    def reset(self, x0: np.ndarray):
        self._vel_cmd = x0[3:6].copy()
