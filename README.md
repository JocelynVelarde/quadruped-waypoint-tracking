# Quadruped Waypoint Trajectory Tracking

A waypoint-based trajectory follower for quadruped robots, built on top of
[Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC).

If you're Nezih please refer to `/results` for a markdown file with the graph analysis

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/0b270739-5ee0-4f1d-a4e5-2c54bc8628dc" />


## What this does

Implements:
- Three test trajectories: square, figure-8, and line-with-stops
- A waypoint follower that converts (x, y, yaw) waypoints into velocity
  commands using P-control with heading wrap-around and tolerance-based
  advancement
- Integration with Quadruped-PyMPC's MPC + gait scheduler so the robot
  actually walks the trajectories
- Tracking metrics (cross-track RMS, heading error, completion time)
  and XY trajectory plots

## Repository layout

    src/                          Waypoint follower & trajectory library
    simulation/run_waypoints.py   Main runner script
    results/                      Generated plots
    docs/                         Written analysis
    third_party/Quadruped-PyMPC/  Walking-controller dependency (submodule)

## Setup

### 1. Clone with submodules

    git clone --recursive https://github.com/<your-username>/<repo-name>.git
    cd <repo-name>

If you forgot `--recursive`:

    git submodule update --init --recursive

### 2. Install Quadruped-PyMPC

Follow the [Quadruped-PyMPC install guide](
https://github.com/iit-DLSLab/Quadruped-PyMPC/blob/main/README_install.md).
Specifically:

    cd third_party/Quadruped-PyMPC
    conda env create -f installation/mamba/nvidia_cuda/mamba_environment.yml
    # (or installation/mamba/integrated_gpu/ if you don't have NVIDIA)
    conda activate quadruped_pympc_env

    cd quadruped_pympc/acados
    mkdir -p build && cd build
    cmake -DACADOS_WITH_SYSTEM_BLASFEO:BOOL=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
    make install -j4
    pip install -e ../interfaces/acados_template
    cd ../../..
    pip install -e .

Then add to your `~/.bashrc`:

    export ACADOS_SOURCE_DIR="$HOME/<path>/third_party/Quadruped-PyMPC/quadruped_pympc/acados"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib"

### 3. Run

    cd ../..    # back to repo root
    python simulation/run_waypoints.py --trajectory line
    python simulation/run_waypoints.py --trajectory square --max-duration 120
    python simulation/run_waypoints.py --trajectory figure8 --max-duration 180

Plots are saved to `results/`.

## Credits

This project depends on:
- **[Quadruped-PyMPC](https://github.com/iit-DLSLab/Quadruped-PyMPC)** by
  Turrisi, Ordonez et al. — provides the MPC, gait scheduler, swing planner,
  and MuJoCo simulation environment.

The waypoint follower, trajectory library, integration runner, and analysis
in this repository are this project's contribution.
