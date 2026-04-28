#!/usr/bin/env python3
"""Waypoint tracking using planar PMP / LQG / MPC body-level controllers.

The professor's controllers (PMP, LQG, MPC) are applied to the planar
body state [px, py, yaw, vx, vy, wz] to compute velocity commands
[vx_cmd, vy_cmd, wz_cmd]. Quadruped-PyMPC's locomotion stack then
realizes those commands through the trot gait.

This is the "same logic as an inverted pendulum" — the controller
stabilizes the body toward the waypoint reference, and the walking
stack handles the legs.

Usage:
    python simulation/run_planar_controllers.py --controller lqg --trajectory line
    python simulation/run_planar_controllers.py --controller mpc --trajectory line
    python simulation/run_planar_controllers.py --controller pmp --trajectory line
    python simulation/run_planar_controllers.py --controller all --trajectory line --no-render
"""

import argparse
import copy
import os
import pathlib
import sys
import time

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

from src.planar_dynamics import PlanarBodyDynamics
from src.planar_controller_lqg import PlanarLQGController
from src.planar_controller_mpc import PlanarMPCController
from src.planar_controller_pmp import PlanarPMPController
from src.waypoint_trajectory import WaypointFollower, FollowerConfig, summarize_tracking
from src.trajectory_library import get_trajectory, TRAJECTORIES


# ─── Controller factory ───────────────────────────────────────────────
def build_controller(name: str, dyn: PlanarBodyDynamics,
                     v_max: float, w_max: float):
    """Build the requested planar controller."""
    name = name.lower()
    if name == "lqg":
        return PlanarLQGController(
            A=dyn.A, B=dyn.B, dt=dyn.dt,
            v_max=v_max, w_max=w_max,
        )
    elif name == "mpc":
        return PlanarMPCController(
            A=dyn.A, B=dyn.B, dt=dyn.dt,
            N=15, v_max=v_max, w_max=w_max,
        )
    elif name == "pmp":
        return PlanarPMPController(
            A=dyn.A, B=dyn.B, dt=dyn.dt,
            v_max=v_max, w_max=w_max,
        )
    else:
        raise ValueError(f"Unknown controller: {name}")


# ─── Velocity smoother ────────────────────────────────────────────────
class VelocitySmoother:
    """Low-pass filter on velocity commands to avoid abrupt changes."""
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
        self._cmd = np.zeros(3)

    def step(self, raw: np.ndarray) -> np.ndarray:
        self._cmd = self.alpha * self._cmd + (1 - self.alpha) * raw
        return self._cmd.copy()

    def reset(self):
        self._cmd = np.zeros(3)


# ─── Main simulation ──────────────────────────────────────────────────
def run_one(qpympc_cfg, controller_name: str, trajectory_name: str,
            max_duration_s: float, render: bool, seed: int,
            output_dir: pathlib.Path):

    np.random.seed(seed)
    print(f"\n  [{controller_name.upper()}] trajectory='{trajectory_name}'")

    # Build waypoints and follower (used for waypoint advancement only)
    waypoints = get_trajectory(trajectory_name)

    # Use a simple P-follower just for waypoint advancement logic
    # The actual velocity COMMAND comes from the planar controller below
    follower_cfg = FollowerConfig(
        v_max=0.5, w_max=0.6,
        pos_tol=0.25, yaw_tol=np.deg2rad(20),
        waypoint_timeout=40.0, decel_radius=0.6,
    )
    follower = WaypointFollower(waypoints, follower_cfg)

    # Build planar dynamics and controller
    ctrl_dt = qpympc_cfg.mpc_params["dt"]   # 0.02s — same cadence as MPC
    dyn = PlanarBodyDynamics(dt=ctrl_dt)
    controller = build_controller(controller_name, dyn, v_max=0.5, w_max=0.6)
    smoother = VelocitySmoother(alpha=0.75)

    # Build env
    robot_name     = qpympc_cfg.robot
    hip_height     = qpympc_cfg.hip_height
    scene_name     = qpympc_cfg.simulation_params["scene"]
    simulation_dt  = qpympc_cfg.simulation_params["dt"]

    env = QuadrupedEnv(
        robot=robot_name, scene=scene_name, sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray((0.0, 1.0)) * hip_height,
        ref_base_ang_vel=(-0.4, 0.4),
        ground_friction_coeff=(0.5, 1.0),
        base_vel_command_type="human",
        state_obs_names=tuple([]),
    )
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant
    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate(
            (env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))
    env.reset(random=False)
    if render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    # Tau limits
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft,
    )
    legs_order = ["FL", "FR", "RL", "RR"]
    feet_traj_geom_ids = None
    feet_GRF_geom_ids  = LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)

    # Quadruped-PyMPC wrapper (unchanged — handles gait, swing, torques)
    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=(
            "ref_base_height", "ref_base_angles", "ref_feet_pos",
            "nmpc_GRFs", "nmpc_footholds",
            "swing_time", "phase_signal", "lift_off_positions",
        ),
    )

    # Logging
    log = {
        "t": [], "x": [], "y": [], "yaw": [],
        "ref_vx": [], "ref_vy": [], "ref_wz": [],
        "actual_vx": [], "actual_vy": [], "actual_wz": [],
        "target_idx": [], "cross_track": [], "heading_err": [],
    }

    RENDER_FREQ = 30
    n_steps     = int(max_duration_s / simulation_dt)
    ctrl_every  = max(1, round(ctrl_dt / simulation_dt))  # controller cadence
    last_render = time.time()
    fell        = False

    # Current velocity command (updated at controller cadence)
    ref_lin = np.zeros(3)
    ref_ang = np.zeros(3)

    # Initialise controller with starting state
    base_pos_init    = env.base_pos.copy()
    base_ori_init    = env.base_ori_euler_xyz.copy()
    base_lin_v_init  = env.base_lin_vel(frame="world")
    base_ang_v_init  = env.base_ang_vel(frame="base")
    x_init = dyn.state_from_env(base_pos_init, base_ori_init,
                                  base_lin_v_init, base_ang_v_init)
    controller.reset(x_init)

    t_wall_start = time.time()
    try:
        for step_i in tqdm(range(n_steps),
                           desc=f"{controller_name.upper()}-{trajectory_name}"):

            # ── Read env state ────────────────────────────────────────
            feet_pos        = env.feet_pos(frame="world")
            feet_vel        = env.feet_vel(frame="world")
            hip_pos         = env.hip_positions(frame="world")
            base_lin_vel    = env.base_lin_vel(frame="world")
            base_ang_vel    = env.base_ang_vel(frame="base")
            base_ori        = env.base_ori_euler_xyz
            base_pos        = copy.deepcopy(env.base_pos)
            com_pos         = copy.deepcopy(env.com)

            t_sim = env.simulation_time
            yaw   = float(base_ori[2])

            # ── Build planar state ────────────────────────────────────
            x_planar = dyn.state_from_env(base_pos, base_ori,
                                           base_lin_vel, base_ang_vel)

            # ── Advance waypoint follower (determines WHICH waypoint) ─
            # We call the follower to track the current target waypoint
            # and get the nominal velocity direction toward it
            nominal_lin, nominal_ang = follower.step(base_pos, yaw, t_sim)
            if follower.is_finished():
                print(f"\n  ✓ Trajectory completed at t={t_sim:.2f}s")
                break

            # ── Planar controller computes velocity command ───────────
            if step_i % ctrl_every == 0:
                tgt = follower.current_target()
                if tgt is not None:
                    # Reference: position = waypoint, velocity = nominal follower cmd
                    x_ref = dyn.get_reference(
                        wp_x=tgt.x, wp_y=tgt.y, wp_yaw=tgt.yaw,
                        cmd_vx=float(nominal_lin[0]),
                        cmd_vy=float(nominal_lin[1]),
                        cmd_wz=float(nominal_ang[2]),
                    )

                    # Call the planar controller
                    if controller_name == "lqg":
                        raw_cmd = controller.step(x_planar, x_ref)
                    else:  # mpc, pmp
                        raw_cmd = controller.compute_control(x_planar, x_ref)

                    # Smooth to avoid abrupt velocity changes
                    smoothed = smoother.step(raw_cmd)

                    ref_lin = np.array([smoothed[0], smoothed[1], 0.0])
                    ref_ang = np.array([0.0, 0.0, smoothed[2]])

            # ── Quadruped-PyMPC computes joint torques ─────────────────
            # Exactly as in run_waypoints.py — unchanged
            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()
            else:
                inertia = qpympc_cfg.inertia.flatten()

            qpos, qvel      = env.mjData.qpos, env.mjData.qvel
            legs_qvel_idx   = env.legs_qvel_idx
            legs_qpos_idx   = env.legs_qpos_idx
            joints_pos      = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR,
                                       RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)
            legs_mass_matrix  = env.legs_mass_matrix
            legs_qfrc_bias    = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive
            feet_jac      = env.feet_jacobians(frame="world", return_rot_jac=False)
            feet_jac_dot  = env.feet_jacobians_dot(frame="world", return_rot_jac=False)

            tau = quadrupedpympc_wrapper.compute_actions(
                com_pos, base_pos, base_lin_vel, base_ori, base_ang_vel,
                feet_pos, hip_pos, joints_pos, None, legs_order,
                simulation_dt, ref_lin, ref_ang, env.step_num,
                qpos, qvel, feet_jac, feet_jac_dot, feet_vel,
                legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix,
                legs_qpos_idx, legs_qvel_idx, tau, inertia, env.mjData.contact,
            )

            for leg in legs_order:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR
            _, _, is_terminated, is_truncated, _ = env.step(action=action)

            # ── Log ───────────────────────────────────────────────────
            log["t"].append(t_sim)
            log["x"].append(float(base_pos[0]))
            log["y"].append(float(base_pos[1]))
            log["yaw"].append(yaw)
            log["ref_vx"].append(float(ref_lin[0]))
            log["ref_vy"].append(float(ref_lin[1]))
            log["ref_wz"].append(float(ref_ang[2]))
            log["actual_vx"].append(float(base_lin_vel[0]))
            log["actual_vy"].append(float(base_lin_vel[1]))
            log["actual_wz"].append(float(base_ang_vel[2]))
            log["target_idx"].append(follower.current_idx)
            log["cross_track"].append(
                follower.log_cross_track[-1] if follower.log_cross_track else 0.0)
            log["heading_err"].append(
                follower.log_heading_err[-1] if follower.log_heading_err else 0.0)

            # ── Render ────────────────────────────────────────────────
            if render and (time.time() - last_render > 1.0 / RENDER_FREQ
                           or env.step_num == 1):
                ctrl_obs = quadrupedpympc_wrapper.get_obs()
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                    swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=ctrl_obs["swing_time"][0], FR=ctrl_obs["swing_time"][1],
                        RL=ctrl_obs["swing_time"][2], RR=ctrl_obs["swing_time"][3],
                    ),
                    lift_off_positions=ctrl_obs["lift_off_positions"],
                    nmpc_footholds=ctrl_obs["nmpc_footholds"],
                    ref_feet_pos=ctrl_obs["ref_feet_pos"],
                    early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )
                tgt = follower.current_target()
                if tgt is not None:
                    render_sphere(env.viewer, [tgt.x, tgt.y, 0.05],
                                  diameter=0.12, color=[1.0, 0.6, 0.0, 0.8],
                                  geom_id=-1)
                for leg_name in legs_order:
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer, vector=feet_GRF[leg_name],
                        pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )
                env.render()
                last_render = time.time()

            if is_terminated or is_truncated:
                print(f"\n  ⚠ {controller_name.upper()} fell at t={t_sim:.2f}s")
                fell = True
                break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()

    elapsed = time.time() - t_wall_start
    metrics = summarize_tracking(follower)
    metrics["wall_time_s"]   = round(elapsed, 2)
    metrics["sim_time_s"]    = round(log["t"][-1], 2) if log["t"] else 0.0
    metrics["fell"]          = fell

    print(f"\n  ─── {controller_name.upper()} summary ───")
    for k, v in metrics.items():
        print(f"    {k:<24s}: {v}")

    if output_dir and log["t"]:
        _save_plot(log, waypoints, controller_name, trajectory_name,
                   output_dir / f"planar_{controller_name}_{trajectory_name}.png")

    return log, waypoints, metrics


# ─── Plotting ─────────────────────────────────────────────────────────
def _save_plot(log, waypoints, ctrl_name, traj_name, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{ctrl_name.upper()} planar controller — {traj_name}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(log["x"], log["y"], "b-", lw=1.8, label="actual path")
    ax.plot([w.x for w in waypoints], [w.y for w in waypoints],
            "r--o", lw=1.0, ms=8, label="waypoints")
    for i, w in enumerate(waypoints):
        ax.annotate(str(i+1), (w.x, w.y), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color="red")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory"); ax.axis("equal")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(log["t"], np.rad2deg(log["yaw"]), "b-", lw=1.0)
    ax.set_xlabel("time [s]"); ax.set_ylabel("yaw [deg]")
    ax.set_title("Heading"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(log["t"], log["ref_vx"],    "r--", lw=1.0, label="ref vx")
    ax.plot(log["t"], log["actual_vx"], "b-",  lw=0.9, alpha=0.8, label="actual vx")
    ax.plot(log["t"], log["ref_vy"],    "m--", lw=1.0, label="ref vy")
    ax.plot(log["t"], log["actual_vy"], "g-",  lw=0.9, alpha=0.8, label="actual vy")
    ax.set_xlabel("time [s]"); ax.set_ylabel("velocity [m/s]")
    ax.set_title("Velocity tracking"); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(log["t"], log["cross_track"], "b-", lw=1.0, label="cross-track [m]")
    ax2 = ax.twinx()
    ax2.plot(log["t"], np.rad2deg(log["heading_err"]),
             "r-", lw=1.0, alpha=0.7, label="heading err [deg]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("cross-track [m]", color="b")
    ax2.set_ylabel("heading err [deg]", color="r")
    ax.set_title("Tracking errors"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {path}")


def _save_comparison(runs, traj_name, waypoints, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Controller comparison — {traj_name}", fontsize=14)
    colors = {"pmp": "tab:red", "lqg": "tab:green", "mpc": "tab:blue"}

    ax = axes[0]
    for name, (log, _, m) in runs.items():
        label = f"{name.upper()}" + (" (fell)" if m.get("fell") else "")
        ax.plot(log["x"], log["y"], lw=1.8,
                color=colors.get(name, "k"), label=label)
    ax.plot([w.x for w in waypoints], [w.y for w in waypoints],
            "k--o", lw=1.0, ms=7, label="waypoints", alpha=0.7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory"); ax.axis("equal")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    for name, (log, _, _) in runs.items():
        ax.plot(log["t"], log["actual_vx"], lw=1.0,
                color=colors.get(name, "k"), label=name.upper(), alpha=0.85)
    ax.axhline(y=0.5, color="k", ls="--", lw=0.8, label="ref vx")
    ax.set_xlabel("time [s]"); ax.set_ylabel("vx [m/s]")
    ax.set_title("Forward velocity tracking")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[2]
    for name, (log, _, _) in runs.items():
        ax.plot(log["t"], log["cross_track"], lw=1.0,
                color=colors.get(name, "k"), label=name.upper(), alpha=0.85)
    ax.set_xlabel("time [s]"); ax.set_ylabel("cross-track error [m]")
    ax.set_title("Path deviation"); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Comparison plot saved: {path}")


# ─── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", default="lqg",
                        choices=["pmp", "lqg", "mpc", "all"])
    parser.add_argument("--trajectory", default="line",
                        choices=list(TRAJECTORIES.keys()))
    parser.add_argument("--max-duration", type=float, default=90.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from quadruped_pympc import config as cfg

    output_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Planar Controller Waypoint Tracking")
    print(f"  Controller(s): {args.controller}")
    print(f"  Trajectory:    {args.trajectory}")
    print(f"  Robot:         {cfg.robot}")
    print(f"  Gait:          {cfg.simulation_params['gait']}")
    print(f"{'=' * 60}")

    controllers = ["pmp", "lqg", "mpc"] if args.controller == "all" else [args.controller]
    runs = {}
    waypoints_used = None

    for ctrl in controllers:
        try:
            log, wps, metrics = run_one(
                qpympc_cfg=cfg,
                controller_name=ctrl,
                trajectory_name=args.trajectory,
                max_duration_s=args.max_duration,
                render=not args.no_render,
                seed=args.seed,
                output_dir=output_dir,
            )
            runs[ctrl] = (log, wps, metrics)
            waypoints_used = wps
        except Exception as e:
            print(f"\n  ✗ {ctrl.upper()} failed: {e}")
            import traceback; traceback.print_exc()

    if len(runs) > 1 and waypoints_used:
        _save_comparison(runs, args.trajectory, waypoints_used,
                         output_dir / f"planar_comparison_{args.trajectory}.png")

        print(f"\n{'=' * 60}")
        print(f"  Metrics comparison — {args.trajectory}")
        print(f"{'=' * 60}")
        keys = ["fell", "cross_track_rms_m", "cross_track_max_m",
                "heading_err_rms_deg", "waypoints_reached", "sim_time_s"]
        hdr = f"  {'metric':<24s} " + " ".join(f"{c.upper():>10s}" for c in runs)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for k in keys:
            row = f"  {k:<24s} "
            for c in runs:
                v = runs[c][2].get(k, "—")
                row += f" {v!s:>10}" if isinstance(v, bool) else (
                    f" {v:>10.3f}" if isinstance(v, (int, float)) else f" {v!s:>10}")
            print(row)

    print("\n  Done.\n")
