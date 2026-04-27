#!/usr/bin/env python3
"""Run Quadruped-PyMPC following a waypoint trajectory (no keyboard).

This is a sibling to simulation/simulation.py. The MPC, gait scheduler,
swing planner and physics step are unchanged — we just replace the
ref_base_lin_vel / ref_base_ang_vel that simulation.py would otherwise
read from the keyboard with the output of a WaypointFollower.

Usage:
    python simulation/simulation_waypoints.py --trajectory square
    python simulation/simulation_waypoints.py --trajectory figure8
    python simulation/simulation_waypoints.py --trajectory line   --no-render
    python simulation/simulation_waypoints.py --trajectory square --max-duration 90

Outputs a plot under results/waypoints_<trajectory>.png with the actual
XY path overlaid on the planned waypoints, plus tracking metrics in the
terminal.
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

# Make sibling imports work whether run from repo root or simulation/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.mujoco.visual import render_sphere, render_vector
from gym_quadruped.utils.quadruped_utils import LegsAttr
from tqdm import tqdm

from quadruped_pympc.helpers.quadruped_utils import plot_swing_mujoco
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

from src.waypoint_trajectory import (
    WaypointFollower, FollowerConfig, summarize_tracking,
)
from src.trajectory_library import get_trajectory, TRAJECTORIES


# ──────────────────────────────────────────────────────────────────────
def run_waypoint_simulation(
    qpympc_cfg,
    trajectory_name: str = "square",
    max_duration_s: float = 90.0,
    render: bool = True,
    seed: int = 0,
    output_dir: pathlib.Path = None,
):
    """Run one simulation that tracks a waypoint trajectory."""
    np.set_printoptions(precision=3, suppress=True)
    np.random.seed(seed)

    # --- Build the waypoint follower BEFORE the env (fail fast on bad name) ---
    waypoints = get_trajectory(trajectory_name)
    follower = WaypointFollower(waypoints, FollowerConfig())
    print(f"\n  Waypoint trajectory '{trajectory_name}' — "
          f"{len(waypoints)} waypoints")

    # --- Pull config (mirrors simulation.py) ---
    robot_name = qpympc_cfg.robot
    hip_height = qpympc_cfg.hip_height
    scene_name = qpympc_cfg.simulation_params["scene"]
    simulation_dt = qpympc_cfg.simulation_params["dt"]

    # --- Build the env. base_vel_command_type doesn't matter here because
    # we override the velocity command every step, but we keep "human" so the
    # internals are happy.
    state_obs_names = []
    env = QuadrupedEnv(
        robot=robot_name,
        scene=scene_name,
        sim_dt=simulation_dt,
        ref_base_lin_vel=np.asarray((0.0, 1.0)) * hip_height,
        ref_base_ang_vel=(-0.4, 0.4),
        ground_friction_coeff=(0.5, 1.0),
        base_vel_command_type="human",
        state_obs_names=tuple(state_obs_names),
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

    # --- Initialize controller-side variables (verbatim from simulation.py) ---
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    tau_soft_limits_scalar = 0.9
    tau_limits = LegsAttr(
        FL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FL] * tau_soft_limits_scalar,
        FR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.FR] * tau_soft_limits_scalar,
        RL=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RL] * tau_soft_limits_scalar,
        RR=env.mjModel.actuator_ctrlrange[env.legs_tau_idx.RR] * tau_soft_limits_scalar,
    )

    feet_traj_geom_ids, feet_GRF_geom_ids = None, LegsAttr(FL=-1, FR=-1, RL=-1, RR=-1)
    legs_order = ["FL", "FR", "RL", "RR"]

    if qpympc_cfg.simulation_params["visual_foothold_adaptation"] != "blind":
        from gym_quadruped.sensors.heightmap import HeightMap
        resolution_heightmap = 0.04
        heightmaps = LegsAttr(
            **{leg: HeightMap(num_rows=7, num_cols=7,
                               dist_x=resolution_heightmap, dist_y=resolution_heightmap,
                               mj_model=env.mjModel, mj_data=env.mjData)
               for leg in ["FL", "FR", "RL", "RR"]}
        )
    else:
        heightmaps = None

    quadrupedpympc_observables_names = (
        "ref_base_height", "ref_base_angles", "ref_feet_pos",
        "nmpc_GRFs", "nmpc_footholds",
        "swing_time", "phase_signal", "lift_off_positions",
    )
    quadrupedpympc_wrapper = QuadrupedPyMPC_Wrapper(
        initial_feet_pos=env.feet_pos,
        legs_order=tuple(legs_order),
        feet_geom_id=env._feet_geom_id,
        quadrupedpympc_observables_names=quadrupedpympc_observables_names,
    )

    # --- Logs ---
    log = {
        "t": [], "x": [], "y": [], "yaw": [],
        "ref_vx": [], "ref_vy": [], "ref_wz": [],
        "actual_vx": [], "actual_vy": [], "actual_wz": [],
        "target_idx": [], "cross_track": [], "heading_err": [],
    }

    RENDER_FREQ = 30
    n_steps = int(max_duration_s / simulation_dt)
    last_render_time = time.time()
    print(f"  sim_dt={simulation_dt}s, max_steps={n_steps}, "
          f"max_duration={max_duration_s}s\n")

    t_start_wall = time.time()
    try:
        for step_i in tqdm(range(n_steps), desc=f"Tracking '{trajectory_name}'"):
            # ── State readout (verbatim from simulation.py) ────────────
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

            # ── ⭐ THE INTEGRATION POINT ⭐ ────────────────────────────
            # In the original simulation.py this line was:
            #     ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel()
            # We override it with our waypoint follower:
            ref_base_lin_vel, ref_base_ang_vel = follower.step(base_pos, yaw, t_sim)

            if follower.is_finished():
                print(f"\n  ✓ Trajectory completed at t={t_sim:.2f}s")
                break
            # ──────────────────────────────────────────────────────────

            # ── Inertia (verbatim from simulation.py) ──────────────────
            if qpympc_cfg.simulation_params["use_inertia_recomputation"]:
                inertia = env.get_base_inertia().flatten()
            else:
                inertia = qpympc_cfg.inertia.flatten()

            qpos, qvel = env.mjData.qpos, env.mjData.qvel
            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx
            joints_pos = LegsAttr(FL=legs_qvel_idx.FL, FR=legs_qvel_idx.FR,
                                   RL=legs_qvel_idx.RL, RR=legs_qvel_idx.RR)

            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame='world', return_rot_jac=False)

            # ── MPC computes torques ───────────────────────────────────
            tau = quadrupedpympc_wrapper.compute_actions(
                com_pos, base_pos, base_lin_vel, base_ori_euler_xyz, base_ang_vel,
                feet_pos, hip_pos, joints_pos, heightmaps, legs_order,
                simulation_dt, ref_base_lin_vel, ref_base_ang_vel, env.step_num,
                qpos, qvel, feet_jac, feet_jac_dot, feet_vel,
                legs_qfrc_passive, legs_qfrc_bias, legs_mass_matrix,
                legs_qpos_idx, legs_qvel_idx, tau, inertia, env.mjData.contact,
            )
            for leg in ["FL", "FR", "RL", "RR"]:
                tau_min, tau_max = tau_limits[leg][:, 0], tau_limits[leg][:, 1]
                tau[leg] = np.clip(tau[leg], tau_min, tau_max)

            # ── Apply action ───────────────────────────────────────────
            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau.FL
            action[env.legs_tau_idx.FR] = tau.FR
            action[env.legs_tau_idx.RL] = tau.RL
            action[env.legs_tau_idx.RR] = tau.RR
            state, _, is_terminated, is_truncated, _ = env.step(action=action)

            # ── Log ────────────────────────────────────────────────────
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

            # ── Render (verbatim from simulation.py) ───────────────────
            if render and (time.time() - last_render_time > 1.0 / RENDER_FREQ
                           or env.step_num == 1):
                _, _, feet_GRF = env.feet_contact_state(ground_reaction_forces=True)
                feet_traj_geom_ids = plot_swing_mujoco(
                    viewer=env.viewer,
                    swing_traj_controller=quadrupedpympc_wrapper.wb_interface.stc,
                    swing_period=quadrupedpympc_wrapper.wb_interface.stc.swing_period,
                    swing_time=LegsAttr(
                        FL=quadrupedpympc_wrapper.get_obs()["swing_time"][0],
                        FR=quadrupedpympc_wrapper.get_obs()["swing_time"][1],
                        RL=quadrupedpympc_wrapper.get_obs()["swing_time"][2],
                        RR=quadrupedpympc_wrapper.get_obs()["swing_time"][3],
                    ),
                    lift_off_positions=quadrupedpympc_wrapper.get_obs()["lift_off_positions"],
                    nmpc_footholds=quadrupedpympc_wrapper.get_obs()["nmpc_footholds"],
                    ref_feet_pos=quadrupedpympc_wrapper.get_obs()["ref_feet_pos"],
                    early_stance_detector=quadrupedpympc_wrapper.wb_interface.esd,
                    geom_ids=feet_traj_geom_ids,
                )

                # Show the current waypoint as a coloured sphere
                tgt = follower.current_target()
                if tgt is not None:
                    render_sphere(
                        viewer=env.viewer,
                        position=[tgt.x, tgt.y, 0.05],
                        diameter=0.10,
                        color=[1.0, 0.6, 0.0, 0.8],  # orange
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

            # ── Reset on failure (we DO NOT reset the follower so it keeps trying) ──
            if is_terminated or is_truncated:
                if is_terminated:
                    print(f"\n  ⚠ Environment terminated at t={t_sim:.2f}s — robot fell")
                    break  # don't pretend success after a fall
                env.reset(random=False)
                quadrupedpympc_wrapper.reset(initial_feet_pos=env.feet_pos(frame="world"))

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        env.close()

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start_wall
    metrics = summarize_tracking(follower)
    metrics["wall_time_s"] = round(elapsed, 2)
    metrics["sim_time_s"] = round(log["t"][-1], 2) if log["t"] else 0.0

    print(f"\n  ─── Tracking summary: {trajectory_name} ───")
    for k, v in metrics.items():
        print(f"    {k:<22s}: {v}")

    # ── Plot ───────────────────────────────────────────────────────────
    if output_dir is not None and log["t"]:
        plot_path = output_dir / f"waypoints_{trajectory_name}.png"
        _plot_run(log, waypoints, trajectory_name, plot_path)
        print(f"\n  Plot saved: {plot_path}")

    return log, metrics


# ──────────────────────────────────────────────────────────────────────
def _plot_run(log, waypoints, trajectory_name, outpath):
    """Save the standard 4-panel diagnostic plot."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Waypoint tracking — {trajectory_name}", fontsize=14)

    # Panel 1: XY path overlay (the headline plot)
    ax = axes[0, 0]
    ax.plot(log["x"], log["y"], "b-", lw=1.8, label="actual path")
    wp_x = [w.x for w in waypoints]; wp_y = [w.y for w in waypoints]
    ax.plot(wp_x, wp_y, "r--o", lw=1.0, ms=8, label="waypoints")
    for i, w in enumerate(waypoints):
        ax.annotate(str(i + 1), (w.x, w.y),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, color="red")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory")
    ax.axis("equal"); ax.grid(True, alpha=0.3); ax.legend()

    # Panel 2: yaw over time
    ax = axes[0, 1]
    ax.plot(log["t"], np.rad2deg(log["yaw"]), "b-", lw=1.0, label="actual")
    ax.set_xlabel("time [s]"); ax.set_ylabel("yaw [deg]")
    ax.set_title("Heading"); ax.grid(True, alpha=0.3); ax.legend()

    # Panel 3: ref vs actual velocity (linear, body-x and body-y)
    ax = axes[1, 0]
    ax.plot(log["t"], log["ref_vx"], "r--", lw=1.0, label="ref vx")
    ax.plot(log["t"], log["actual_vx"], "b-", lw=0.9, alpha=0.8, label="actual vx")
    ax.plot(log["t"], log["ref_vy"], "m--", lw=1.0, label="ref vy")
    ax.plot(log["t"], log["actual_vy"], "g-", lw=0.9, alpha=0.8, label="actual vy")
    ax.set_xlabel("time [s]"); ax.set_ylabel("velocity [m/s]")
    ax.set_title("Linear velocity tracking")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    # Panel 4: tracking errors
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory", default="square",
                        choices=list(TRAJECTORIES.keys()))
    parser.add_argument("--max-duration", type=float, default=90.0,
                        help="Maximum simulation time in seconds")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from quadruped_pympc import config as cfg

    output_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Quadruped-PyMPC — Waypoint Trajectory Tracking")
    print(f"  Trajectory:  {args.trajectory}")
    print(f"  Max time:    {args.max_duration}s")
    print(f"  Render:      {not args.no_render}")
    print(f"  Robot:       {cfg.robot}")
    print(f"  Gait:        {cfg.simulation_params['gait']}")
    print(f"{'=' * 60}")

    run_waypoint_simulation(
        qpympc_cfg=cfg,
        trajectory_name=args.trajectory,
        max_duration_s=args.max_duration,
        render=not args.no_render,
        seed=args.seed,
        output_dir=output_dir,
    )

    print("\n  Done.\n")
