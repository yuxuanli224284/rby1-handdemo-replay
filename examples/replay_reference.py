"""Reference loader for the RBY1 replay files (SDK-agnostic). It reads one joint CSV, checks the robot's current joint state
against the first row, and produces a dense, smooth command stream at your controller rate. Sending the commands to the robot is
left to your own code (see send_command below), because it depends on your control stack (rby1-sdk, ROS, ...).

  python replay_reference.py ../trajectories/joint/swap_v1_ep2_s0.5_joint.csv --rate 100 [--dry-run]

Requirements: numpy, scipy.
"""
import argparse, csv
import numpy as np
from scipy.interpolate import PchipInterpolator

START_TOL_RAD = 0.02   # how close the robot must be to the first row before starting (about 1.1 deg); adjust to your needs


def load(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["time_s"]) for r in rows])
    joint_cols = [c for c in rows[0] if c.endswith(("_joint_1", "_joint_2", "_joint_3", "_joint_4", "_joint_5", "_joint_6", "_joint_7")) and "gripper" not in c]
    q = np.array([[float(r[c]) for c in joint_cols] for r in rows])
    grip_open_m = np.array([float(r["left_gripper_open_m"]) for r in rows])        # per-finger opening, 0 = closed .. 0.05 = fully open
    grip_closed = np.array([int(r["gripper_closed"]) for r in rows]).astype(bool)
    phase = [r["phase"] for r in rows]
    return t, joint_cols, q, grip_open_m, grip_closed, phase


def dense(t, q, grip_open_m, grip_closed, rate):
    """shape-preserving (PCHIP) interpolation of the joints (no overshoot between samples); the gripper command is held
    (closed from the first closed sample on, never reopens)"""
    tg = np.arange(0.0, t[-1] + 1e-9, 1.0 / rate)
    qg = PchipInterpolator(t, q, axis=0)(tg)
    k = np.clip(np.searchsorted(t, tg, side="right") - 1, 0, len(t) - 1)
    return tg, qg, grip_open_m[k], grip_closed[k]


def send_command(t, names, q, grip_open_m, grip_closed):
    """REPLACE with your robot interface (position command of all joints in `names`, gripper target)."""
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--rate", type=float, default=100.0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t, names, q, go, gc, phase = load(a.csv)
    print(f"{a.csv}: {len(t)} samples, {t[-1]:.1f} s, joints {len(names)}; close command at t = {t[np.argmax(gc)]:.2f} s; phases {sorted(set(phase), key=phase.index)}")
    tg, qg, gog, gcg = dense(t, q, go, gc, a.rate)
    v = np.abs(np.diff(qg, axis=0)) * a.rate
    print(f"dense stream: {len(tg)} steps at {a.rate:g} Hz, peak joint speed {v.max():.3f} rad/s")
    # current = read_joint_state()            # <- your code: dict name -> rad
    # dev = max(abs(current[n] - q[0, i]) for i, n in enumerate(names)); assert dev < START_TOL_RAD, "move the robot to the start posture first"
    if a.dry_run: return
    for k in range(len(tg)): send_command(tg[k], names, qg[k], gog[k], gcg[k])


if __name__ == "__main__":
    main()
