#!/usr/bin/env python3
"""Walk waypoint trajectories using the professor's body-level controllers.

This is the integration of:
- Track 1: Quadruped-PyMPC's locomotion stack (gait scheduler + swing planner)
- Track 2: Professor's PMP / LQG / MPC body-level controllers

The professor's controllers compute the body wrench / GRFs (the "brain" — what
forces should the trunk exert through the feet to track the velocity command).
Quadruped-PyMPC's WBInterface handles everything else:
- Periodic gait generator (which feet are stance vs swing right now)
- Foothold planner (Raibert heuristic for where to place feet)
- Swing trajectory generator (Bezier curves through the air)
- Stance torque mapping (Jacobian transpose for stance feet)
- Joint torque assembly (final torques sent to the robot)

Usage:
    python simulation/run_walk_with_prof_controller.py --controller lqg --trajectory line
    python simulation/run_walk_with_prof_controller.py --controller mpc --trajectory square
    python simulation/run_walk_with_prof_controller.py --controller pmp --trajectory figure8 --max-duration 60
    python simulation/run_walk_with_prof_controller.py --controller all --trajectory line --no-render
"""

import argparse
import copy
import os
import pathlib
import sys
import time
from pprint import pprint

import numpy as np
import mujoco

# Make sibling imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.interfaces.wb_interface import WBInterface

from src.body_controller_adapter import BodyControllerAdapter
from src.waypoint_trajectory import (
    WaypointFollower, FollowerConfig, summarize_tracking,
)
from src.trajectory_library import get_trajectory, TRAJECTORIES


# ──────────────────────────────────────────────────────────────────────
def run_one(
    qpympc_cfg,
    controller_name: str,
    trajectory_name: str,
    max_duration_s: float,
    render: bool,
    seed: int,
    output_dir: pathlib.Path,
):
    """Run one (controller, trajectory) experiment."""
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    waypoints = get_trajectory(trajectory_name)
    follower = WaypointFollower(waypoints, FollowerConfig())
    print(f"\n  Trajectory '{trajectory_name}': {len(waypoints)} waypoints")

    # --- Pull config (mirrors simulation.py) ---
    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    # --- Build env ---
    env = QuadrupedEnv(
        robot=robot_name, scene=scene_name, sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray((0.0, 1.0)) * hip_height,
        ref_base_ang_vel=(-0.4, 0.4),
        ground_friction_coeff=(0.5, 1.0),
        base_vel_command_type="human",
        state_obs_names=tuple([]),
    )
    pprint(env.get_hyperparameters())
    env.mjModel.opt.gravity[2] = -qpympc_cfg.gravity_constant
    if qpympc_cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate(
            (env.mjModel.qpos0[:7], qpympc_cfg.qpos0_js))

    env.reset(random=False)
    if render:
        env.render()
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    # --- Tau setup (verbatim from simulation.py) ---
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    feet_traj_geom_ids = None
    feet_GRF_geom_ids = LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]
    heightmaps = None  # blind mode

    # --- ⭐ Build the WBInterface (locomotion stack) ---
    # We use this directly instead of QuadrupedPyMPC_Wrapper so we can splice
    # in the professor's controller between the state-update step and the
    # torque-mapping step.
    wb_interface = WBInterface(
        initial_feet_pos=env.feet_pos(frame="world"),
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
    )

    # --- ⭐ Build the body-level controller (the professor's brain) ---
    body_adapter = BodyControllerAdapter(
        controller_name=controller_name,
        mass=qpympc_cfg.mass,
        inertia=qpympc_cfg.inertia,
        hip_height=hip_height,
        dt=qpympc_cfg.mpc_params["dt"],
    )

    # --- Logs ---
    log = {
        "t": [], "x": [], "y": [], "yaw": [],
        "ref_vx": [], "ref_vy": [], "ref_wz": [],
        "actual_vx": [], "actual_vy": [], "actual_wz": [],
        "target_idx": [], "cross_track": [], "heading_err": [],
        "grf_norm": [],
    }

    RENDER_FREQ = 30
    n_steps = int(max_duration_s / simulation_dt)
    last_render_time = time.time()
    print(f"  [{controller_name.upper()}] sim_dt={simulation_dt}s, "
          f"max_steps={n_steps}, max_duration={max_duration_s}s\n")

    # State for the orchestrated loop
    nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3),
                         RL=np.zeros(3), RR=np.zeros(3))
    nmpc_footholds = LegsAttr(FL=np.zeros(3), FR=np.zeros(3),
                              RL=np.zeros(3), RR=np.zeros(3))
    nmpc_predicted_state = np.zeros(12)
    best_sample_freq = wb_interface.pgg.step_freq
    optimize_swing = 0

    mpc_frequency = qpympc_cfg.simulation_params["mpc_frequency"]

    t_start_wall = time.time()
    fell = False
    try:
        for step_i in tqdm(range(n_steps),
                           desc=f"{controller_name.upper()}-{trajectory_name}"):
            # ── State readout ─────────────────────────────────────────
            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame='world')
            hip_pos = env.hip_positions(frame="world")
            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz
            base_pos = copy.deepcopy(env.base_pos)
            com_pos = copy.deepcopy(env.com)

            t_sim = env.simulation_time
            yaw = float(base_ori_euler_xyz[2])

            # ── Waypoint follower → velocity command ────────────────
            ref_base_lin_vel, ref_base_ang_vel = follower.step(base_pos, yaw, t_sim)
            if follower.is_finished():
                print(f"\n  ✓ Trajectory completed at t={t_sim:.2f}s")
                break

            # ── Inertia ──────────────────────────────────────────────
            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()
            else:
                inertia = qpympc_cfg.inertia.flatten()

            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx
            joints_pos = LegsAttr(
                FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR,
                RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR,
            )

            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)

            # ────────────────────────────────────────────────────────
            #  STEP 1: Update state and reference (Quadruped-PyMPC)
            # ────────────────────────────────────────────────────────
            state_current, ref_state, contact_sequence, step_height, optimize_swing = (
                wb_interface.update_state_and_reference(
                    com_pos, base_pos, base_lin_vel,
                    base_ori_euler_xyz, base_ang_vel,
                    feet_pos, hip_pos, joints_pos,
                    heightmaps, legs_order, simulation_dt,
                    ref_base_lin_vel, ref_base_ang_vel,
                    env.mjData.contact,
                )
            )

            # ────────────────────────────────────────────────────────
            #  STEP 2: ⭐ Body-level control (PROFESSOR'S CONTROLLER)
            # ────────────────────────────────────────────────────────
            if step_i % round(1 / (mpc_frequency * simulation_dt)) == 0:
                nmpc_GRFs, nmpc_footholds = body_adapter.compute_control(
                    state_current, ref_state, contact_sequence,
                    inertia=inertia,
                )

            # ────────────────────────────────────────────────────────
            #  STEP 3: Stance + swing torque mapping (Quadruped-PyMPC)
            # ────────────────────────────────────────────────────────
            # Dummy joints (not used in non-kinodynamic mode)
            empty_joints = LegsAttr(FL=np.zeros(3), FR=np.zeros(3),
                                    RL=np.zeros(3), RR=np.zeros(3))

            tau, _, _ = wb_interface.compute_stance_and_swing_torque(
                simulation_dt, qpos, qvel,
                feet_jac, feet_jac_dot,
                feet_pos, feet_vel,
                legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix,
                nmpc_GRFs, nmpc_footholds,
                legs_qpos_idx, legs_qvel_idx,
                tau, optimize_swing, best_sample_freq,
                empty_joints, empty_joints, empty_joints,
                nmpc_predicted_state,
                env.mjData.contact,
            )

            # ── Torque limits ──────────────────────────────────────
            for leg in legs_order:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            # ── Apply action ───────────────────────────────────────
            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR
            state, _, is_terminated, is_truncated, _ = env.step(action=action)

            # ── Log ────────────────────────────────────────────────
            grf_norm = float(
                np.linalg.norm(nmpc_GRFs.FL) + np.linalg.norm(nmpc_GRFs.FR)
                + np.linalg.norm(nmpc_GRFs.RL) + np.linalg.norm(nmpc_GRFs.RR)
            )
            log["t"].append(t_sim)
            log["x"].append(float(base_pos[0]))
            log["y"].append(float(base_pos[1]))
            log["yaw"].append(yaw)
            log["ref_vx"].append(float(ref_base_lin_vel[0]))
            log["ref_vy"].append(float(ref_base_lin_vel[1]))
            log["ref_wz"].append(float(ref_base_ang_vel[2]))
            log["actual_vx"].append(float(base_lin_vel[0]))
            log["actual_vy"].append(float(base_lin_vel[1]))
            log["actual_wz"].append(float(base_ang_vel[2]))
            log["target_idx"].append(follower.current_idx)
            log["cross_track"].append(
                follower.log_cross_track[-1] if follower.log_cross_track else 0.0)
            log["heading_err"].append(
                follower.log_heading_err[-1] if follower.log_heading_err else 0.0)
            log["grf_norm"].append(grf_norm)

            # ── Render ─────────────────────────────────────────────
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ
                           or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=wb_interface.stc,
                    swing_period=wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=wb_interface.stc.swing_time[0],
                        FR=wb_interface.stc.swing_time[1],
                        RL=wb_interface.stc.swing_time[2],
                        RR=wb_interface.stc.swing_time[3],
                    ),
                    lift_off_positions=wb_interface.frg.lift_off_positions,
                    nmpc_footholds=nmpc_footholds,
                    ref_feet_pos=LegsAttr(
                        FL=ref_state["ref_foot_FL"].reshape(3, 1),
                        FR=ref_state["ref_foot_FR"].reshape(3, 1),
                        RL=ref_state["ref_foot_RL"].reshape(3, 1),
                        RR=ref_state["ref_foot_RR"].reshape(3, 1),
                    ),
                    early_stance_detector=wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                # Show current waypoint as orange sphere
                tgt = follower.current_target()
                if tgt is not None:
                    render_sphere(
                        viewer=env.viewer,
                        position=[tgt.x, tgt.y, 0.05],
                        diameter=0.12,
                        color=[1.0, 0.6, 0.0, 0.8],
                        geom_id=-1,
                    )

                for leg_id, leg_name in enumerate(legs_order):
                    feet_GRF_geom_ids[leg_name] = render_vector(
                        env.viewer,
                        vector=feet_GRF[leg_name], pos=feet_pos[leg_name],
                        scale=np.linalg.norm(feet_GRF[leg_name]) * 0.005,
                        color=np.array([0, 1, 0, 0.5]),
                        geom_id=feet_GRF_geom_ids[leg_name],
                    )
                env.render()
                last_render_time = time.time()

            # ── Failure handling ───────────────────────────────────
            if is_terminated or is_truncated:
                print(f"\n  ⚠ {controller_name.upper()}: robot fell at t={t_sim:.2f}s")
                fell = True
                break

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start_wall
    metrics = summarize_tracking(follower)
    metrics["wall_time_s"] = round(elapsed, 2)
    metrics["sim_time_s"] = round(log["t"][-1], 2) if log["t"] else 0.0
    metrics["fell"] = fell
    metrics["mean_grf_norm"] = float(np.mean(log["grf_norm"])) if log["grf_norm"] else 0.0

    print(f"\n  ─── {controller_name.upper()} on '{trajectory_name}' ───")
    for k, v in metrics.items():
        print(f"    {k:<22s}: {v}")

    # ── Plot ─────────────────────────────────────────────────────────
    if output_dir is not None and log["t"]:
        plot_path = output_dir / f"prof_{controller_name}_{trajectory_name}.png"
        _plot_run(log, waypoints, controller_name, trajectory_name, plot_path)
        print(f"\n  Plot saved: {plot_path}")

    return log, waypoints, metrics


# ──────────────────────────────────────────────────────────────────────
def _plot_run(log, waypoints, controller_name, trajectory_name, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{controller_name.upper()} body-level + walking — {trajectory_name}",
                 fontsize=14)

    ax = axes[0, 0]
    ax.plot(log["x"], log["y"], "b-", lw=1.8, label="actual path")
    wp_x = [w.x for w in waypoints]
    wp_y = [w.y for w in waypoints]
    ax.plot(wp_x, wp_y, "r--o", lw=1.0, ms=8, label="waypoints")
    for i, w in enumerate(waypoints):
        ax.annotate(str(i + 1), (w.x, w.y),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="red")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory"); ax.axis("equal")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(log["t"], np.rad2deg(log["yaw"]), "b-", lw=1.0, label="actual")
    ax.set_xlabel("time [s]"); ax.set_ylabel("yaw [deg]")
    ax.set_title("Heading"); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 0]
    ax.plot(log["t"], log["ref_vx"], "r--", lw=1.0, label="ref vx")
    ax.plot(log["t"], log["actual_vx"], "b-", lw=0.9, alpha=0.8, label="actual vx")
    ax.plot(log["t"], log["ref_vy"], "m--", lw=1.0, label="ref vy")
    ax.plot(log["t"], log["actual_vy"], "g-", lw=0.9, alpha=0.8, label="actual vy")
    ax.set_xlabel("time [s]"); ax.set_ylabel("velocity [m/s]")
    ax.set_title("Linear velocity tracking")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(log["t"], log["cross_track"], "b-", lw=1.0, label="cross-track [m]")
    ax2 = ax.twinx()
    ax2.plot(log["t"], np.rad2deg(log["heading_err"]),
             "r-", lw=1.0, alpha=0.7, label="heading err [deg]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("cross-track [m]", color="b")
    ax2.set_ylabel("heading err [deg]", color="r")
    ax.set_title("Tracking errors"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
def _plot_comparison(runs, trajectory_name, waypoints, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Body-level controller comparison on '{trajectory_name}'",
                 fontsize=14)

    ax = axes[0]
    colors = {"pmp": "tab:red", "lqg": "tab:green", "mpc": "tab:blue"}
    for name, (log, _, metrics) in runs.items():
        label = name.upper()
        if metrics.get("fell"):
            label += " (fell)"
        ax.plot(log["x"], log["y"], lw=1.6, color=colors.get(name, "k"), label=label)
    wp_x = [w.x for w in waypoints]; wp_y = [w.y for w in waypoints]
    ax.plot(wp_x, wp_y, "k--o", lw=1.0, ms=8, label="waypoints", alpha=0.7)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory"); ax.axis("equal")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    for name, (log, _, _) in runs.items():
        ax.plot(log["t"], log["grf_norm"], lw=1.0,
                color=colors.get(name, "k"), label=name.upper(), alpha=0.85)
    ax.set_xlabel("time [s]"); ax.set_ylabel("Σ ||GRF|| [N]")
    ax.set_title("Total control effort"); ax.grid(True, alpha=0.3); ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", default="lqg",
                        choices=["pmp", "lqg", "mpc", "all"])
    parser.add_argument("--trajectory", default="line",
                        choices=list(TRAJECTORIES.keys()))
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from quadruped_pympc import config as cfg

    output_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Walk + Professor's body-level controllers")
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
            log, waypoints, metrics = run_one(
                qpympc_cfg=cfg,
                controller_name=ctrl,
                trajectory_name=args.trajectory,
                max_duration_s=args.max_duration,
                render=not args.no_render,
                seed=args.seed,
                output_dir=output_dir,
            )
            runs[ctrl] = (log, waypoints, metrics)
            waypoints_used = waypoints
        except Exception as e:
            print(f"\n  ✗ {ctrl.upper()} crashed: {e}")
            import traceback
            traceback.print_exc()

    # Comparison plot
    if len(runs) > 1 and waypoints_used is not None:
        comp_path = output_dir / f"prof_comparison_{args.trajectory}.png"
        _plot_comparison(runs, args.trajectory, waypoints_used, comp_path)
        print(f"\n  Comparison plot saved: {comp_path}")

        # Side-by-side metrics table
        print(f"\n{'=' * 60}")
        print(f"  Comparison — '{args.trajectory}'")
        print(f"{'=' * 60}")
        keys = ["fell", "cross_track_rms_m", "cross_track_max_m",
                "heading_err_rms_deg", "waypoints_reached",
                "sim_time_s", "mean_grf_norm"]
        header = f"  {'metric':<22s} " + " ".join(f"{c.upper():>10s}" for c in runs)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for k in keys:
            row = f"  {k:<22s} "
            for c in runs:
                v = runs[c][2].get(k, "—")
                row += f" {v!s:>10}" if isinstance(v, bool) else (
                    f" {v:>10.3f}" if isinstance(v, (int, float)) else f" {v!s:>10}"
                )
            print(row)

    print("\n  Done.\n")
