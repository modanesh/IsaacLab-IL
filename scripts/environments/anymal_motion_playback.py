"""anymal_motion_playback.py playback of a motion clip on the ANYmal-D.

Two modes:

KINEMATIC (default) base pinned, joint states written directly every step;
physics used only for rendering. Nothing is filtered, scaled or PD-tracked, so
what you see is EXACTLY what is stored in the clip. This is the ground truth for
verifying the *retarget geometry* (do the legs trace the right gait?).

PHYSICS (--physics) base is FREE (gravity + ground contact on); the clip joint
angles are fed as PD position *targets* through the ANYmal-D actuator model, and
the base moves however the dynamics dictate. This answers a different question:
is the motion dynamically FEASIBLE would the robot actually stay upright
executing it, or does it topple? Falls are detected (base height / tilt) and
reported per loop, and the robot re-initializes each loop so you can watch
repeatedly. Use this to triage clips that look fine kinematically but are
dynamically broken (discard or fix them before BC/RL).

Note: the retargeted clips have ~zero root translation (base was pinned during
retargeting), so in physics mode the robot steps roughly in place a fall here
means the in-place gait itself is unstable, which is exactly what we want to
catch. It is open-loop reference tracking, NOT a trained policy.

Supported clip formats (auto-detected):
  - retarget.py output:  uses 'joint_pos_isaac' (absolute, Isaac order)
  - training format:     'joint_pos' (PyBullet order, residuals around the
                         default pose, SIGN_FLIP applied) reconstructed via
                         q = joint_pos[PYB_TO_ISAAC] * SIGN_FLIP + default

Usage:
    ./isaaclab.sh -p scripts/environments/anymal_motion_playback.py \
        --motion scripts/environments/data/pace.npz [--speed 1.0] [--physics]
"""
import argparse
import os
import sys

import numpy as np

PYB_TO_ISAAC = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=int)
SIGN_FLIP = np.ones(12, dtype=np.float32)
SIGN_FLIP[2] = -1.0
SIGN_FLIP[3] = -1.0
ISAAC_DEFAULT = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.4, -0.4, 0.4, -0.4, -0.8, 0.8, -0.8, 0.8],
    dtype=np.float32,
)

parser = argparse.ArgumentParser(description="ANYmal-D motion playback in Isaac Lab.")
parser.add_argument("--motion", type=str, required=True, help="Path to the motion .npz file.")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (<1 = slow motion).")
parser.add_argument("--robot", type=str, default="anymal_d")
parser.add_argument("--physics", action="store_true",
                    help="Free base + gravity + contacts; drive clip joints as PD targets and "
                         "report falls (dynamic feasibility). Default is kinematic (base pinned).")
parser.add_argument("--fall-height", type=float, default=0.28,
                    help="[physics] base height (m) below which the robot is counted as fallen.")
parser.add_argument("--fall-tilt", type=float, default=0.6,
                    help="[physics] fall if projected-gravity z-component exceeds -this "
                         "(0=upright -> -1; -0.6 ~= 53deg tilt).")
parser.add_argument("--hold-frame", type=int, default=-1,
                    help="Hold a SINGLE clip frame statically (base grounded, joints written, "
                         "GUI open) so you can position the camera and screenshot it for a figure.")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# After the app is running we can import Isaac Lab modules.
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retargeting"))
from helpers import get_robot_cfg  # noqa: E402


@configclass
class PlaybackSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = None


def load_clip(npz_path: str, robot_joint_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (joint_pos_isaac (T,12), root_pos (T,3), root_rot (T,4) wxyz, frame_dt)."""
    data = dict(np.load(npz_path, allow_pickle=True))
    print(f"[Playback] keys in npz: {sorted(data.keys())}")
    frame_dt = float(data.get("frame_duration", data.get("dt", 1.0 / 30.0)))

    if "joint_pos_isaac" in data:
        q = np.asarray(data["joint_pos_isaac"], dtype=np.float32)
        # Guard against articulation-order drift between export and playback.
        if "joint_names" in data:
            clip_names = [str(s) for s in data["joint_names"]]
            if clip_names != robot_joint_names:
                perm = [clip_names.index(n) for n in robot_joint_names]
                q = q[:, perm]
                print(f"[Playback] reordered clip joints to articulation order: {perm}")
        print("[Playback] using absolute Isaac-order 'joint_pos_isaac'")
    elif "joint_pos" in data:
        raw_pyb = np.asarray(data["joint_pos"], dtype=np.float32)
        default = np.asarray(
            data.get("isaac_default_joint_pos", ISAAC_DEFAULT), dtype=np.float32
        )
        q = raw_pyb[:, PYB_TO_ISAAC] * SIGN_FLIP[None, :] + default[None, :]
        print("[Playback] reconstructed absolute joints from training-format 'joint_pos' "
              "(pyb order, residual, SIGN_FLIP)")
    else:
        raise KeyError(f"No 'joint_pos_isaac' or 'joint_pos' in {npz_path}")

    T = q.shape[0]
    root_pos = np.asarray(data.get("root_pos"), dtype=np.float32) if "root_pos" in data else None
    root_rot = np.asarray(data.get("root_rot"), dtype=np.float32) if "root_rot" in data else None
    if root_pos is None or root_pos.shape != (T, 3):
        root_pos = np.tile(np.array([[0.0, 0.0, 0.5]], dtype=np.float32), (T, 1))
    if root_rot is None or root_rot.shape != (T, 4):
        root_rot = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (T, 1))

    print(f"[Playback] {T} frames, dt={frame_dt:.4f}s ({T * frame_dt:.1f}s)")
    return q, root_pos, root_rot, frame_dt


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.0, 1.6], target=[0.0, 0.0, 0.45])

    scene_cfg = PlaybackSceneCfg(num_envs=100, env_spacing=2.0)
    scene_cfg.robot = get_robot_cfg(args_cli.robot)
    if args_cli.physics or args_cli.hold_frame >= 0:
        # Put the ground at z=0 so the feet actually land on it (kinematic mode
        # keeps it out of the way at -1.05 since the base is pinned there).
        scene_cfg.ground.init_state = AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    scene.update(sim_cfg.dt)
    robot.update(sim_cfg.dt)
    device = args_cli.device

    q_clip, root_pos, root_rot, frame_dt = load_clip(args_cli.motion, list(robot.joint_names))
    T = q_clip.shape[0]
    q_clip_t = torch.tensor(q_clip, dtype=torch.float32, device=device)
    root_pose_t = torch.tensor(np.concatenate([root_pos, root_rot], axis=-1),
                               dtype=torch.float32, device=device)
    zero_vel6 = torch.zeros((1, 6), dtype=torch.float32, device=device)
    zero_vel12 = torch.zeros((1, q_clip.shape[1]), dtype=torch.float32, device=device)

    def interp(sim_time: float) -> tuple[torch.Tensor, int, int]:
        """Loop-interpolated clip joint angles at the given playback time."""
        u = (sim_time * args_cli.speed / frame_dt) % T
        i0 = int(u)
        i1 = (i0 + 1) % T
        alpha = float(u - i0)
        return (1.0 - alpha) * q_clip_t[i0] + alpha * q_clip_t[i1], i0, i1

    if args_cli.hold_frame >= 0:
        run_hold(sim, sim_cfg, scene, robot, q_clip_t, args_cli.hold_frame, device)
    elif args_cli.physics:
        run_physics(sim, sim_cfg, scene, robot, interp, root_pose_t, zero_vel12, T, frame_dt)
    else:
        run_kinematic(sim, sim_cfg, scene, robot, interp, root_pose_t, zero_vel6, zero_vel12, T)

    print("[Playback] done.")


def run_hold(sim, sim_cfg, scene, robot, q_clip_t, frame_idx, device):
    """Hold ONE clip frame statically for a figure screenshot: base grounded and
    pinned, joints written kinematically, GUI open until the window is closed."""
    T = q_clip_t.shape[0]
    fi = max(0, min(int(frame_idx), T - 1))
    qN = q_clip_t[fi]
    zero12 = torch.zeros((1, q_clip_t.shape[1]), dtype=torch.float32, device=device)
    zero6 = torch.zeros((1, 6), dtype=torch.float32, device=device)
    # base height that rests the lowest foot on the ground (uses the robot's own FK)
    h = _ground_robot_height(sim, sim_cfg, scene, robot, qN, zero12, device)
    root_pose = torch.tensor([[0.0, 0.0, h, 1.0, 0.0, 0.0, 0.0]],
                             dtype=torch.float32, device=device)
    # side profile (+x is forward / to the right, matching a horse walking right)
    sim.set_camera_view(eye=[0.05, -3.0, 0.9], target=[0.0, 0.0, 0.45])
    print(f"[Hold] holding frame {fi}/{T - 1} at base height {h:.2f} m. "
          f"Position the camera in the viewport and screenshot. Close the window to stop.")
    while simulation_app.is_running():
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(qN.unsqueeze(0), zero12)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        robot.update(sim_cfg.dt)


def run_kinematic(sim, sim_cfg, scene, robot, interp, root_pose_t, zero_vel6, zero_vel12, T):
    sim_time = 0.0
    print(f"[Playback] KINEMATIC: looping at {args_cli.speed:.2f}x. Close the window to stop.")
    while simulation_app.is_running():
        q, i0, i1 = interp(sim_time)
        robot.write_root_pose_to_sim(root_pose_t[i0].unsqueeze(0))
        robot.write_root_velocity_to_sim(zero_vel6)
        robot.write_joint_state_to_sim(q.unsqueeze(0), zero_vel12)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        robot.update(sim_cfg.dt)
        if i1 < i0:
            print(f"[Playback] loop complete (t={sim_time:.1f}s)")
        sim_time += sim_cfg.dt


def _foot_body_ids(robot) -> list[int]:
    names = list(robot.body_names)
    return [i for i, n in enumerate(names) if n.endswith("_FOOT") or n.endswith("FOOT")]


def _ground_robot_height(sim, sim_cfg, scene, robot, q0, zero12, device) -> float:
    """Spawn at frame-0 pose well above the ground, step once, and read the real
    foot heights via FK to find the base height that puts the lowest foot exactly
    on the ground. Robust to any clip (uses the robot's own geometry, not the
    clip's foot targets). Returns the corrected base height (m)."""
    spawn_h = 0.80
    root = robot.data.default_root_state.clone()
    root[:, 0:3] = torch.tensor([[0.0, 0.0, spawn_h]], device=device)
    root[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    root[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(q0.unsqueeze(0), zero12)
    robot.set_joint_position_target(q0.unsqueeze(0))
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_cfg.dt)
    robot.update(sim_cfg.dt)
    foot_ids = _foot_body_ids(robot)
    if not foot_ids:
        return 0.60
    foot_z = robot.data.body_pos_w[0, foot_ids, 2]
    lowest = float(foot_z.min().item())
    # base must rise so the lowest foot sits ~1 cm above ground, then settles on
    return spawn_h - lowest + 0.01


def run_physics(sim, sim_cfg, scene, robot, interp, root_pose_t, zero_vel12, T, frame_dt):
    device = args_cli.device
    q0 = interp(0.0)[0]
    zero12 = zero_vel12

    # Init height from the robot's ACTUAL foot geometry at the start pose. The
    # earlier fixed guess (clip root 0.5 + 0.03) put the feet ~5 cm UNDERGROUND
    # (feet sit ~0.58 m below the base), so PhysX launched the robot on frame 0
    # and every clip "fell" instantly a setup bug, not a bad motion.
    init_h = _ground_robot_height(sim, sim_cfg, scene, robot, q0, zero12, device)

    def reset_robot():
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = torch.tensor([[0.0, 0.0, init_h]], device=device)
        root[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        root[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(q0.unsqueeze(0), zero12)
        robot.set_joint_position_target(q0.unsqueeze(0))
        scene.write_data_to_sim()
        robot.update(sim_cfg.dt)

    def settle(seconds: float = 0.5) -> tuple[float, float]:
        """Hold the frame-0 pose so the robot stabilizes from the spawn transient
        before the gait starts. Returns (final height, final tilt gz)."""
        for _ in range(max(1, int(seconds / sim_cfg.dt))):
            robot.set_joint_position_target(q0.unsqueeze(0))
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
            robot.update(sim_cfg.dt)
        return (float(robot.data.root_pos_w[0, 2].item()),
                float(robot.data.projected_gravity_b[0, 2].item()))

    def fallen() -> tuple[bool, float, float]:
        h = float(robot.data.root_pos_w[0, 2].item())
        gz = float(robot.data.projected_gravity_b[0, 2].item())  # -1 upright -> 0 tipped
        return (h < args_cli.fall_height or gz > -args_cli.fall_tilt), h, gz

    reset_robot()
    print(f"[Playback] PHYSICS: free base + gravity, clip joints as PD targets. "
          f"grounded init height {init_h:.2f} m. Fall if h<{args_cli.fall_height} m or tilt gz>-{args_cli.fall_tilt}.")
    h_s, gz_s = settle(0.5)
    print(f"[Playback] settled at start pose: height {h_s:.2f} m, tilt gz {gz_s:+.2f} "
          f"{'(stable)' if (h_s>=args_cli.fall_height and gz_s<=-args_cli.fall_tilt) else '(ALREADY UNSTABLE at rest!)'}")
    print(f"[Playback] NOTE: open-loop tracking has NO balance feedback. Even a perfect gait "
          f"can topple here; this is a RELATIVE feasibility/longevity check, not a verdict.\n"
          f"           Looping at {args_cli.speed:.2f}x. Close the window to stop.")

    sim_time = 0.0
    fell = False
    fell_t = 0.0
    min_h = float("inf")
    loop_idx = 0
    while simulation_app.is_running():
        q, i0, i1 = interp(sim_time)
        # Drive joints as POSITION TARGETS through the actuator model; the base
        # falls/balances under physics. No root or joint state is written.
        robot.set_joint_position_target(q.unsqueeze(0))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)
        robot.update(sim_cfg.dt)

        is_fall, h, gz = fallen()
        min_h = min(min_h, h)
        if is_fall and not fell:
            fell = True
            fell_t = sim_time
            print(f"  >>> FELL during loop {loop_idx} at t={sim_time:.2f}s "
                  f"(height {h:.2f} m, tilt gz {gz:+.2f}) <<<")

        if i1 < i0:  # clip wrapped -> end of a loop
            surv = f"survived {fell_t:.2f}s then fell" if fell else "survived FULL loop upright"
            print(f"[Playback] loop {loop_idx}: {surv} | min base height {min_h:.2f} m")
            loop_idx += 1
            fell = False
            min_h = float("inf")
            reset_robot()
            settle(0.3)
            sim_time = 0.0
            continue
        sim_time += sim_cfg.dt


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    # Isaac Sim's simulation_app.close() can hang the interpreter; the playback
    # is already finished (you closed the window), so flush and force-exit
    # instead of waiting on a graceful close that may never return.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
