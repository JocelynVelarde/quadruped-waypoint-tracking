"""Body-level controller adapter.

Wraps the professor's PMP / LQG / MPC controllers (which assume all-feet-stance,
12-dim GRF output) so they fit Quadruped-PyMPC's pipeline (state dict in,
LegsAttr GRFs out).

Architecture:
    Quadruped-PyMPC's WBInterface produces state dict + ref dict + contact seq
                                     ↓
    BodyControllerAdapter translates to professor's 12-vec format
                                     ↓
    Professor's PMP / LQG / MPC computes 12 GRFs (body-level wrench distribution)
                                     ↓
    BodyControllerAdapter reshapes back to LegsAttr
                                     ↓
    Quadruped-PyMPC's WBInterface uses these GRFs in stance-torque mapping,
    plans swing trajectories, and produces final joint torques.

The professor's controllers do NOT compute footholds — those still come from
Quadruped-PyMPC's Raibert generator. We pass them through unchanged.
"""

from __future__ import annotations
import numpy as np
from gym_quadruped.utils.quadruped_utils import LegsAttr

from .dynamics import QuadrupedDynamics
from .controller_pmp import PontryaginController
from .controller_lqg import LQGController
from .controller_mpc import MPCController


# Default Q/R weights — tuned for walking, not stationary posture.
# Lower z-position weight because the body bobs at gait frequency.
# Lower lateral velocity weight because the body sways during trot.
DEFAULT_Q = np.diag([
    0.1,  0.1,  30,      # position: tiny x,y cost, moderate z (height)
    15,  0.1,  10,      # velocity: track vx, ignore vy (tiny), moderate vz
    20,   20,   0.1,     # orientation: roll+pitch, tiny yaw (not zero!)
    0.1,  0.1,  0.1,     # angular velocity: tiny everywhere
])

DEFAULT_R = np.eye(12) * 5e-5   # allow larger forces


class BodyControllerAdapter:
    """Drop-in replacement for Quadruped-PyMPC's SRBDControllerInterface.compute_control.

    Usage:
        adapter = BodyControllerAdapter(
            controller_name="lqg",
            mass=12.5, inertia=robot_inertia,
            hip_height=0.225, dt=0.01,
        )

        # In simulation loop:
        nmpc_GRFs, nmpc_footholds = adapter.compute_control(
            state_current, ref_state, contact_sequence)
    """

    def __init__(
        self,
        controller_name: str,
        mass: float,
        inertia: np.ndarray,
        hip_height: float,
        dt: float = 0.01,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        mpc_horizon: int = 10,
        pmp_horizon: int = 100,
        force_limit: float = 200.0,
        

    ):
        self.name = controller_name.lower()
        self.dt = dt
        self.hip_height = hip_height
        self.force_limit = force_limit
        

        Q = Q if Q is not None else DEFAULT_Q
        R = R if R is not None else DEFAULT_R
        Q_f = Q * 5.0

        # Build dynamics with foot offsets matching Quadruped-PyMPC's nominal stance.
        self.dyn = QuadrupedDynamics(mass=mass, inertia=inertia, dt=dt)
        # The default foot offsets in QuadrupedDynamics may not match this robot —
        # try to use canonical mini-cheetah/aliengo shape; fine-tuning per robot
        # would require reading hip positions from the env.
        self.dyn.r_feet_body = np.array([
            [ 0.179,  0.111, -0.310],   # FL
            [ 0.179, -0.111, -0.310],   # FR
            [-0.214,  0.111, -0.310],   # RL
            [-0.214, -0.111, -0.310],   # RR
        ])
        # Reference standing state (used for linearization)
        x_ref0 = self.dyn.standing_state(height=0.341)  # actual standing height
        self.u_ref_standing = self.dyn.standing_control()

        # Linearize once at standing — we'll re-use this linearization
        # throughout. (For production-quality control we'd re-linearize each
        # step at the current reference, but that's expensive and not needed
        # for the assignment.)
        Ad, Bd, gd = self.dyn.get_linear_system(x_ref0)
        Ac, Bc = self.dyn.continuous_AB(x_ref0)

        # Build the requested controller
        if self.name == "pmp":
            self.controller = PontryaginController(
                A=Ac, B=Bc, Q_s=Q, R_u=R, Q_f=Q_f,
                g_aff=self.dyn.gravity_vector() / dt,
                dt=dt, horizon=pmp_horizon,
            )
            # PMP needs to be solved once to populate gains.
            # We use steady-state mode in compute_control (no step_idx), so the
            # discrete-sweep solution isn't strictly required, but solving it
            # ensures K_ss is computed from the same data path.
            try:
                self.controller.solve_discrete_sweep(x_ref0.copy(), x_ref0)
            except Exception as e:
                print(f"  [adapter] PMP discrete-sweep failed: {e} (using steady-state K)")

        elif self.name == "lqg":
            self.controller = LQGController(
                A_d=Ad, B_d=Bd, g_d=gd,
                Q=Q * dt, R=R * dt,
                Q_proc=np.diag([1e-3]*3 + [1e-2]*3 + [5e-3]*3 + [1e-2]*3),
                R_meas=np.diag([5e-3]*3 + [2e-2]*3 + [1e-2]*3 + [5e-2]*3),
            )
            self.controller.set_initial_estimate(x_ref0),
            self._use_kalman = True

        elif self.name == "mpc":
            self.controller = MPCController(
                A_d=Ad, B_d=Bd, g_d=gd,
                Q=Q * dt, R=R * dt, Q_f=Q_f * dt,
                N=mpc_horizon, mu=0.6, fz_max=force_limit,
            )

        else:
            raise ValueError(f"Unknown controller '{controller_name}'. "
                             f"Use 'pmp', 'lqg', or 'mpc'.")

        print(f"  [adapter] Built {self.name.upper()} body-level controller")
        self._ref_vel_smooth = np.zeros(3)
        

    # ------------------------------------------------------------------
    def compute_control(
        self,
        state_current: dict,
        ref_state: dict,
        contact_sequence: np.ndarray,
        inertia: np.ndarray = None,
        external_wrenches: np.ndarray = None,
    ) -> tuple[LegsAttr, LegsAttr]:
        """Same signature as Quadruped-PyMPC's SRBDControllerInterface.compute_control.

        Returns:
            (nmpc_GRFs, nmpc_footholds) — both LegsAttr of 3-vectors.
            nmpc_footholds is passed through from ref_state (we don't replan footholds).
        """
        
        # --- Translate Quadruped-PyMPC dicts → professor's 12-vector ---
        x      = self._dict_to_state_vec(state_current)
        x_ref  = self._refdict_to_state_vec(ref_state)

        raw_ref_vel = x_ref[3:6].copy()
        alpha = 0.85
        self._ref_vel_smooth = alpha * self._ref_vel_smooth + (1 - alpha) * raw_ref_vel
        x_ref[3:6] = self._ref_vel_smooth

        # Also match position reference to current position
        # so MPC tracks velocity, not absolute position
        x_ref[2] = 0.341   # keep height reference

        # Current contact mask (which feet are in stance right now)
        current_contact = np.array([
            contact_sequence[0][0], contact_sequence[1][0],
            contact_sequence[2][0], contact_sequence[3][0],
        ], dtype=bool)

        # --- Call the professor's controller ---
        try:
            if self.name == "pmp":
                # Use steady-state K (no step_idx) — appropriate for closed-loop walking
                u = self.controller.compute_control(x, x_ref, self.u_ref_standing)

            elif self.name == "lqg":
                # Direct state feedback (skip Kalman; we have ground truth)
                u = self.controller.step(x, x_ref, self.u_ref_standing)

            elif self.name == "mpc":
                # MPC accepts a contact mask natively — pass per-step contact schedule
                # contact_sequence has shape (4, horizon). Transpose to (horizon, 4)
                # for MPC controller's expected (N, 4) format.
                cs = np.asarray(contact_sequence).T.astype(bool)  # (horizon, 4)
                # Trim or pad to MPC horizon
                if cs.shape[0] >= self.controller.N:
                    cs = cs[:self.controller.N]
                else:
                    pad = np.tile(cs[-1:], (self.controller.N - cs.shape[0], 1))
                    cs = np.vstack([cs, pad])
                print(f"  x[3:6]={x[3:6].round(3)}  x_ref[3:6]={x_ref[3:6].round(3)}")
                u = self.controller.compute_control(
                    x, x_ref, self.u_ref_standing, contact_mask=cs)
        except Exception as e:
            print(f"  [adapter] {self.name.upper()} compute_control failed: {e}; "
                  f"using gravity-compensation fallback")
            u = self.u_ref_standing.copy()

        actual_vx = float(state_current["linear_velocity"][0])
        ref_vx = float(self._ref_vel_smooth[0])
        if actual_vx > ref_vx + 0.05:
            brake = max(0.05, ref_vx / max(actual_vx, 0.01))
            u[0] *= brake   # FL fx
            u[3] *= brake   # FR fx
            u[6] *= brake   # RL fx
            u[9] *= brake   # RR fx

        # Strip lateral + symmetrise fz
        u_ref_local = self.u_ref_standing
        for i in range(4):
            u[i*3 + 1] = u_ref_local[i*3 + 1]

        fz_left  = (u[2] + u[8])  / 2.0
        fz_right = (u[5] + u[11]) / 2.0
        fz_mean  = (fz_left + fz_right) / 2.0
        u[2] = u[8] = fz_mean
        u[5] = u[11] = fz_mean

        # Saturate
        u = np.clip(u, -self.force_limit, self.force_limit)

        # --- Reshape 12-vec → LegsAttr ---
        # Apply contact mask (zero out swing feet) — Quadruped-PyMPC also does
        # this downstream, but we do it here too for consistency.
        nmpc_GRFs = LegsAttr(
            FL=u[0:3]   * current_contact[0],
            FR=u[3:6]   * current_contact[1],
            RL=u[6:9]   * current_contact[2],
            RR=u[9:12]  * current_contact[3],
        )

        # --- Footholds: pass through from ref_state (Quadruped-PyMPC's Raibert) ---
        nmpc_footholds = LegsAttr(
            FL=np.asarray(ref_state["ref_foot_FL"]).reshape(3),
            FR=np.asarray(ref_state["ref_foot_FR"]).reshape(3),
            RL=np.asarray(ref_state["ref_foot_RL"]).reshape(3),
            RR=np.asarray(ref_state["ref_foot_RR"]).reshape(3),
        )

        return nmpc_GRFs, nmpc_footholds

    # ------------------------------------------------------------------
    @staticmethod
    def _dict_to_state_vec(state: dict) -> np.ndarray:
        """Quadruped-PyMPC's state dict → 12-vector [p, v, θ, ω]."""
        return np.concatenate([
            np.asarray(state["position"]).flatten(),
            np.asarray(state["linear_velocity"]).flatten(),
            np.asarray(state["orientation"]).flatten(),
            np.asarray(state["angular_velocity"]).flatten(),
        ])

    @staticmethod
    def _refdict_to_state_vec(ref: dict) -> np.ndarray:
        """Quadruped-PyMPC's reference dict → 12-vector [p_ref, v_ref, θ_ref, ω_ref]."""
        return np.concatenate([
            np.asarray(ref["ref_position"]).flatten(),
            np.asarray(ref["ref_linear_velocity"]).flatten(),
            np.asarray(ref["ref_orientation"]).flatten(),
            np.asarray(ref["ref_angular_velocity"]).flatten(),
        ])
