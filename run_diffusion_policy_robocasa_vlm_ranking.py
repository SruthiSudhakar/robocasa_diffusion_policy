"""
VLM-ranked diffusion policy rollout in RoboCasa, using the SIMULATOR as the lookahead model.

This is the simulation analogue of
    diffusion_policy/run_diffusion_policy_blocking_upright_bottle.py
On the real robot that script oversamples action chunks, prunes to N diverse candidates, ships
them to a remote video-generation + VLM-ranking server, blocks on a ranking.json, and executes the
VLM-chosen chunk. Here we have a simulator, so we skip the generative video model entirely:

    1. Capture one observation; oversample action chunks; prune to --num-samples diverse candidates.
    2. SAVE the full sim state s_t.
    3. For each candidate, restore s_t, roll the chunk forward in the sim, render the executed
       sub-steps into a short mp4 (a real video of s_t -> s_{t+1}).
    4. Restore s_t.
    5. Rank the N candidate mp4s with the local Qwen2.5-VL ranker (trl/.../rank_videos.py).
    6. Restore s_t and execute ONLY the winning chunk for real, recording the rollout video.
    7. Loop until success / done / --max_steps.

Runs --num-rollouts sequential rollouts in one job on a single env (reset between each, scene seed
--env-seed + rollout_index, matching the plain eval with num_envs=1), and reports an aggregate
success rate. The per-decision lookahead is the only "parallelism" (over --num-samples candidates).

The VLM ranker needs a newer `transformers` than this env ships, so it cannot run in-process. By
default this script AUTO-LAUNCHES rank_serve_robocasa.py as a subprocess under a VLM-capable env
(--rank-server-python, default the vlmoverlay env) and talks to it over a local job inbox; pass
--no-launch-rank-server to attach to a server you started yourself.

Example (smoke test, one command):
cd /proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=egl python run_diffusion_policy_robocasa_vlm_ranking.py \
    --checkpoint /proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/data/jgd/2026.05.23/23.28.09_train_diffusion_unet_hybrid_target_atomic_seen_jgd_CloseToasterOvenDoor/checkpoints/epoch=0100-train_loss=0.0027.ckpt \
    --task CloseToasterOvenDoor \
    --split target \
    --num-samples 3 \
    --device cuda:1 --rank-server-gpus 0 \
    --num-rollouts 50 \
    --vlm-checkpoint /proj/vondrick3/sruthi/Appaji/trl/outputs/CloseToasterOvenDoor_20260524_170526_robocasa/checkpoint-750

GPUs: keep --rank-server-gpus off the policy's --device. The server holds a 7B model; the policy
its own. With CUDA_VISIBLE_DEVICES=0,1 above, the policy runs on cuda:1 and the ranker on cuda:0.
"""
import sys
# line-buffer stdout/stderr so logs interleave correctly with subprocess/VLM output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import argparse
import collections
import copy
import datetime
import json
import os
import pathlib
import subprocess
import time

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import colored

import robocasa  # noqa: F401  (registers the robocasa/* gym envs)
from robocasa.utils.dataset_registry_utils import get_task_horizon

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.env_runner.robomimic_image_runner import create_env

# Reuse the DDIM swap helper from the plain eval script.
from eval_robocasa import swap_to_ddim

OmegaConf.register_new_resolver("eval", eval, replace=True)

# The Qwen2.5-VL ranker needs a newer `transformers` than this (robocasa) env ships, so it cannot
# run in-process. Instead we drive rank_serve_robocasa.py (in the trl repo, run under the vlmoverlay
# env) via the same atomic-job-inbox + poll-for-ranking.json protocol the bottle script uses with
# its remote server -- only here it is local. The server loads the 7B model once and reuses it.
VLM_SCRIPTS_DIR = "/proj/vondrick3/sruthi/Appaji/trl/examples/scripts/myscripts"
RANK_SERVER_SCRIPT = os.path.join(VLM_SCRIPTS_DIR, "rank_serve_robocasa.py")
RANK_SERVER_PYTHON_DEFAULT = "/proj/vondrick3/sruthi/miniconda3/envs/vlmoverlay/bin/python"


def launch_rank_server(server_python, checkpoint, task_name, gpus, max_pixels, batch_size,
                       inbox, done, error, log_path):
    """Spawn rank_serve_robocasa.py under `server_python` (vlmoverlay env). Returns (proc, logfile)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    n_gpu = max(len([g for g in gpus.split(",") if g.strip()]), 1)
    cmd = [server_python, RANK_SERVER_SCRIPT,
           "--inbox_dir", str(inbox), "--done_dir", str(done), "--error_dir", str(error),
           "--checkpoint", checkpoint, "--task_name", task_name,
           "--gpu_ids", ",".join(str(i) for i in range(n_gpu)),
           "--batch_size", str(batch_size), "--max_pixels", max_pixels]
    logf = open(log_path, "w")
    print(f"Launching rank server (CUDA_VISIBLE_DEVICES={gpus}): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc, logf


def wait_for_server_ready(proc, log_path, timeout):
    """Block until the server logs its 'ready. watching' line (model loaded)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"rank server exited early (code {proc.returncode}); see {log_path}")
        try:
            if "ready. watching" in pathlib.Path(log_path).read_text():
                return
        except FileNotFoundError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"rank server not ready within {timeout}s; see {log_path}")


def submit_rank_job(inbox, name, output_subdir, video_paths, task_name):
    """Write an atomic (tmp -> rename) job json into the server's inbox."""
    job = {"name": name, "output_subdir": str(output_subdir),
           "video_paths": [str(v) for v in video_paths], "task_name": task_name}
    inbox = pathlib.Path(inbox)
    tmp = inbox / f"{name}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(job, f)
    os.replace(tmp, inbox / f"{name}.json")


def wait_for_ranking(output_subdir, poll, timeout, proc=None):
    """Block until ranking.json appears in output_subdir; return the parsed dict."""
    rp = pathlib.Path(output_subdir) / "ranking.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if rp.exists():
            try:
                with open(rp) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass  # half-written; retry (server writes atomically, so this is belt-and-suspenders)
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"rank server died (code {proc.returncode}) while waiting for {rp}")
        time.sleep(poll)
    raise TimeoutError(f"ranking timeout after {timeout}s waiting for {rp}")


def select_diverse_indices(actions_exec, k, terminal_weight=4.0, n_terminal=2,
                           joint_weights=None, seed_idx=0):
    """Greedy farthest-point sampling over terminal-weighted, joint-weighted flattened L2.

    Ported from run_diffusion_policy_blocking_upright_bottle.py; joint_weights defaults to ones
    (RoboCasa's action_dim differs from the 7-DOF arm the bottle script targeted).
    """
    M, T, D = actions_exec.shape
    assert 1 <= k <= M, f'k={k} must be in [1, M={M}]'
    if joint_weights is None:
        joint_weights = np.ones(D, dtype=actions_exec.dtype)
    assert joint_weights.shape == (D,), f'joint_weights shape {joint_weights.shape} != ({D},)'
    w_t = np.ones(T, dtype=actions_exec.dtype)
    w_t[-n_terminal:] = terminal_weight
    w = (np.sqrt(w_t)[:, None]
         * np.sqrt(joint_weights.astype(actions_exec.dtype))[None, :])
    flat = (actions_exec * w[None, :, :]).reshape(M, -1)
    selected = [seed_idx]
    min_d = np.linalg.norm(flat - flat[seed_idx], axis=1)
    min_d[seed_idx] = -np.inf
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        d_new = np.linalg.norm(flat - flat[nxt], axis=1)
        min_d = np.minimum(min_d, d_new)
        min_d[nxt] = -np.inf
    return selected


def parse_args():
    p = argparse.ArgumentParser(
        description="VLM-ranked diffusion policy rollout in RoboCasa (simulator lookahead).")
    # --- shared with run_diffusion_policy_robocasa.py ---
    p.add_argument("-c", "--checkpoint", required=True, help="Path to the trained .ckpt file.")
    p.add_argument("-o", "--output_dir", default=None,
                   help="Where to write the run dir. Defaults to <ckpt_dir>/../evals/<ckpt_stem>/<split>/.")
    p.add_argument("-d", "--device", default="cuda:0",
                   help="Device for the diffusion policy. The VLM ranker always pins cuda:0.")
    p.add_argument("-t", "--task", required=True, help="A single RoboCasa task/env name.")
    p.add_argument("-s", "--split", required=True, choices=["pretrain", "target"])
    p.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"])
    p.add_argument("--num_inference_steps", default=8, type=int)
    p.add_argument("--max_steps", default=None, type=int,
                   help="Real action-step budget for the rollout (default: 1.5x task horizon).")
    p.add_argument("--render_size", default="848x480", type=str,
                   help="WxH for the rendered candidate + rollout videos (even dims for h264).")
    p.add_argument("--render_camera", default=None, type=str,
                   help="Camera for rendering (default: derived from the policy's render_obs_key).")
    p.add_argument("--seed", default=None, type=int,
                   help="Per-run torch seed for the policy's denoising noise (env scene uses --env-seed).")
    p.add_argument("-n", "--num-rollouts", default=50, type=int,
                   help="Number of sequential rollouts in this job (single env, reset between each). "
                        "Rollout i uses scene seed --env-seed + i.")
    p.add_argument("--env-seed", default=100000, type=int,
                   help="Base kitchen-scene seed. Rollout i uses --env-seed + i (matches the plain "
                        "eval's test_start_seed + rollout_index with num_envs=1).")
    # --- ported from the bottle script ---
    p.add_argument("--num-samples", default=5, type=int,
                   help="Number of candidate chunks ranked by the VLM each cycle.")
    p.add_argument("--oversample", default=50, type=int,
                   help="Sample this many chunks per cycle, then prune to --num-samples via greedy FPS. "
                        "0 disables oversampling (==num-samples).")
    p.add_argument("--n-act-exec", default=0, type=int,
                   help="How many actions of each chunk to roll out / execute. 0 = full predicted horizon (pred_h).")
    p.add_argument("--picking-strategy", default="best", choices=["best", "worst", "random"])
    # --- VLM ranker (runs out-of-process via rank_serve_robocasa.py) ---
    p.add_argument("--vlm-checkpoint", default=None,
                   help="Path to the trained Qwen2.5-VL ranker checkpoint. Required unless "
                        "--no-launch-rank-server (then the server you started already has it).")
    p.add_argument("--vlm-task-name", default=None,
                   help="Task name for the ranker prompt (default: --task). The server auto-derives the "
                        "token (e.g. CloseToasterOvenDoor -> [CLOSE_TOASTER_OVEN_DOOR]).")
    p.add_argument("--vlm-max-pixels", default="960x540", type=str, help="WxH image budget for the ranker.")
    p.add_argument("--vlm-batch-size", default=4, type=int, help="Pairwise comparisons per generate() call.")
    # --- rank server lifecycle ---
    p.add_argument("--rank-server-python", default=RANK_SERVER_PYTHON_DEFAULT,
                   help="Python interpreter (env with a VLM-capable transformers) to run the rank server.")
    p.add_argument("--rank-server-gpus", default="0",
                   help="CUDA_VISIBLE_DEVICES for the auto-launched rank server (keep off the policy's GPU).")
    p.add_argument("--no-launch-rank-server", action="store_true",
                   help="Do not auto-launch the server; attach to one already watching --rank-dir/inbox.")
    p.add_argument("--rank-dir", default=None,
                   help="Base dir for the server's inbox/done/error (default: <run_dir>/rank).")
    p.add_argument("--rank-poll-sec", default=2.0, type=float)
    p.add_argument("--rank-timeout-sec", default=1800.0, type=float, help="Per-cycle ranking timeout.")
    p.add_argument("--rank-ready-timeout-sec", default=900.0, type=float,
                   help="How long to wait for the auto-launched server to load the model.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    if args.num_samples < 1:
        sys.exit("--num-samples must be >= 1")
    oversample = args.oversample if args.oversample != 0 else args.num_samples
    if oversample < args.num_samples:
        sys.exit(f"--oversample ({oversample}) must be >= --num-samples ({args.num_samples})")

    try:
        render_w, render_h = (int(x) for x in args.render_size.lower().split("x"))
    except ValueError:
        sys.exit(f"--render_size must be WxH (e.g. 848x480), got: {args.render_size}")
    if render_w % 2 or render_h % 2:
        sys.exit(f"--render_size dims must be even for h264, got: {args.render_size}")

    vlm_task_name = args.vlm_task_name or args.task
    # --num-samples 1 is a no-VLM baseline: no ranking happens, so no server/checkpoint is needed.
    use_vlm = args.num_samples >= 2
    if use_vlm and not args.no_launch_rank_server:
        if not args.vlm_checkpoint:
            sys.exit("--vlm-checkpoint is required unless --no-launch-rank-server is set.")
        if not os.path.exists(args.vlm_checkpoint):
            sys.exit(f"VLM checkpoint not found: {args.vlm_checkpoint}")

    # resolve the per-run policy-noise seed (printable / reproducible)
    seed = args.seed if args.seed is not None else (torch.seed() & 0x7FFFFFFF)
    seed = int(seed)

    # ---- 1. Load checkpoint + policy (mirrors eval_robocasa.eval_task) ----
    print(colored(f"Loading checkpoint: {args.checkpoint}", "cyan"))
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cfg = copy.deepcopy(OmegaConf.to_container(cfg))
    cfg = OmegaConf.create(cfg)

    shape_meta = OmegaConf.to_container(cfg.task.shape_meta)
    er = cfg.task.env_runner
    render_obs_key = er.get("render_obs_key", "robot0_agentview_right_image")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir="/tmp/dp_robocasa_vlm")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model

    if args.sampler == "ddim":
        swap_to_ddim(policy, num_inference_steps=args.num_inference_steps)
    elif args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(colored(f"Per-run policy-noise seed (torch only): {seed}", "cyan"))

    n_obs = int(policy.n_obs_steps)
    action_dim = int(policy.action_dim)
    # We rank / prune / render / execute over the policy's FULL predicted horizon
    # (pred['action_pred']), not just the n_action_steps actionable chunk. The executable future
    # is action_pred[:, n_obs_steps-1:], of length pred_h = horizon - (n_obs_steps - 1).
    pred_h = int(cfg.horizon) - (n_obs - 1)
    n_act_exec = args.n_act_exec if args.n_act_exec != 0 else pred_h
    if not (1 <= n_act_exec <= pred_h):
        sys.exit(f"--n-act-exec ({n_act_exec}) must be in [1, predicted horizon={pred_h}]")
    print(f"Policy: n_obs_steps={n_obs}, horizon={int(cfg.horizon)}, "
          f"action_dim={action_dim}; ranking/rendering over the full {pred_h}-step horizon, "
          f"executing first {n_act_exec}/{pred_h} per cycle")

    horizon = get_task_horizon(task=args.task)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(horizon * 1.5)

    # ---- 2. Output dir ----
    if args.output_dir is None:
        base_output_dir = os.path.join(
            os.path.dirname(args.checkpoint), "../evals",
            os.path.basename(args.checkpoint).replace(".ckpt", ""), args.split)
    else:
        base_output_dir = args.output_dir
    run_stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    run_dir = pathlib.Path(base_output_dir) / f"{args.task}_VLM_{run_stamp}_seed{seed}"
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    print(colored(f"Run dir: {run_dir}", "cyan"))

    # ---- 3. Rank server (out-of-process; loads the VLM once) ----
    rank_base = pathlib.Path(args.rank_dir) if args.rank_dir else (run_dir / "rank")
    rank_inbox = rank_base / "inbox"
    rank_done = rank_base / "done"
    rank_error = rank_base / "error"
    for d in (rank_inbox, rank_done, rank_error):
        d.mkdir(parents=True, exist_ok=True)
    server_proc = None
    if not use_vlm:
        print(colored("--num-samples=1: no-VLM baseline (no rank server, no candidate rollouts).", "cyan"))
    elif args.no_launch_rank_server:
        print(colored(f"Using external rank server watching {rank_inbox}", "cyan"))
    else:
        server_log = run_dir / "rank_server.log"
        server_proc, _ = launch_rank_server(
            args.rank_server_python, args.vlm_checkpoint, vlm_task_name,
            args.rank_server_gpus, args.vlm_max_pixels, args.vlm_batch_size,
            rank_inbox, rank_done, rank_error, server_log)
        print(colored(f"Waiting for rank server to load the model (log: {server_log}) ...", "cyan"))
        wait_for_server_ready(server_proc, server_log, args.rank_ready_timeout_sec)
        print(colored("Rank server ready.", "cyan"))

    # ---- 4. Build a single env (no vector/multistep/video wrappers) ----
    robocasa_env = create_env(split=args.split, env_name=args.task, seed=args.env_seed)
    env = RobomimicImageWrapper(
        env=robocasa_env,
        shape_meta=shape_meta,
        init_state=None,
        render_obs_key=render_obs_key,
        render_width=render_w,
        render_height=render_h,
        render_camera=args.render_camera,
    )
    # env.env is the RoboCasaGymEnv; env.env.env is the underlying robosuite env that actually
    # holds .sim and the Python-side episode counters. We must set those on the robosuite env
    # directly: RoboCasaGymEnv forwards attribute *reads* via __getattr__ but not writes.
    rs_env = env.env.env

    obs_hist = None  # (re)bound at the start of each rollout below

    def save_state():
        return {
            "sim": rs_env.sim.get_state().flatten(),
            "timestep": rs_env.timestep,
            "cur_time": rs_env.cur_time,
            "done": rs_env.done,
        }

    def restore_state(s):
        rs_env.sim.set_state_from_flattened(s["sim"])
        rs_env.sim.forward()
        rs_env.timestep = s["timestep"]
        rs_env.cur_time = s["cur_time"]
        rs_env.done = s["done"]

    def build_obs_batch(batch):
        """Stack obs history -> (batch, n_obs, *feat) torch tensors keyed by shape_meta obs keys."""
        obs_t = {}
        for key in shape_meta["obs"].keys():
            stacked = np.stack([o[key] for o in obs_hist], axis=0)[None, ...]  # (1, n_obs, *feat)
            tiled = np.broadcast_to(stacked, (batch, *stacked.shape[1:])).copy()
            obs_t[key] = torch.from_numpy(tiled.astype(np.float32)).to(device)
        return obs_t

    fps = int(er.get("fps", 10))
    crf = int(er.get("crf", 22))
    # RoboCasa control runs at robosuite_fps (20 Hz). The plain eval records one frame every
    # steps_per_render control steps and encodes at `fps`, so playback is real-time. Match that
    # here -- otherwise we'd write every control step at `fps` and the video would play at half
    # speed (2x duration). See robomimic_image_runner.py:85-86 + video_recording_wrapper.py:38.
    robosuite_fps = int(er.get("robosuite_fps", 20))
    steps_per_render = max(robosuite_fps // fps, 1)

    media_dir = run_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    def run_one_cycle(scene_seed, r_cand_dir, cycle, global_cycle, rollout_rec, base_step):
        """One observe -> oversample -> render candidates -> VLM rank -> execute-winner cycle.

        Returns (winner_idx, votes, kept, n_executed, reached_success, reached_done, success_step_in_cycle).
        Records executed frames into rollout_rec and updates obs_hist in place.
        """
        # Inference: oversample candidate chunks in one batched forward pass.
        torch.manual_seed(seed + global_cycle)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + global_cycle)
        with torch.no_grad():
            pred = policy.predict_action(build_obs_batch(oversample))
        # Full predicted horizon. Drop the (n_obs_steps-1) past-aligned steps so index 0 is "now"
        # (mirrors how the policy slices `action` = action_pred[:, n_obs_steps-1 : +n_action_steps]).
        actions_all = pred["action_pred"][:, n_obs - 1:].detach().cpu().numpy()  # (oversample, pred_h, D)

        # Prune to num_samples diverse candidates over the FULL predicted horizon (same horizon
        # the candidate videos are rendered over), not just the executed window.
        keep = list(range(oversample))
        if oversample > args.num_samples:
            keep = select_diverse_indices(
                actions_all[:, :n_act_exec], k=args.num_samples,
                terminal_weight=4.0, n_terminal=2, seed_idx=0)
        actions_kept = actions_all[keep, :n_act_exec]  # (num_samples, n_act_exec, action_dim)

        # Save s_t, render each candidate into a short mp4 (named <i>.mp4 so the ranker's
        # winner_idx maps straight back to the candidate index) by rolling it forward in sim.
        s_t = save_state()
        cycle_subdir = r_cand_dir / f"cycle{cycle:06d}"
        cycle_subdir.mkdir(parents=True, exist_ok=True)
        np.save(cycle_subdir / "actions.npy", actions_kept)
        if actions_kept.shape[0] < 2:
            # --num-samples 1: no-VLM baseline. The ranker needs >=2 videos, so skip candidate
            # rendering + ranking entirely and just execute the single sampled chunk. The sim is
            # still at s_t (we never rolled a candidate), so execution starts from the right state.
            winner_idx = 0
            votes = None
        else:
            # Render each candidate into a short mp4 (named <i>.mp4 so the ranker's winner_idx maps
            # straight back to the candidate index) by rolling it forward in sim.
            cand_paths = []
            for i in range(actions_kept.shape[0]):
                restore_state(s_t)
                vpath = str(cycle_subdir / f"{i}.mp4")
                rec = VideoRecorder.create_h264(fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf)
                rec.start(vpath)
                for step_i, a in enumerate(actions_kept[i, :], start=1):
                    env.step(a.astype(np.float32))
                    if step_i % steps_per_render == 0:  # downsample 20 Hz -> fps for real-time playback
                        rec.write_frame(env.render())
                rec.stop()
                cand_paths.append(vpath)

            # Restore s_t; sanity-check the revert is exact.
            restore_state(s_t)
            if not np.allclose(rs_env.sim.get_state().flatten(), s_t["sim"]):
                print(colored(f"  cycle {cycle}: WARNING sim state not exactly restored", "red"))

            # Rank candidates: submit a job to the server and block on ranking.json.
            job_name = f"{run_stamp}_seed{scene_seed}_{cycle:06d}"
            submit_rank_job(rank_inbox, job_name, cycle_subdir, cand_paths, vlm_task_name)
            result = wait_for_ranking(cycle_subdir, args.rank_poll_sec,
                                      args.rank_timeout_sec, proc=server_proc)
            votes = result["votes"]
            if args.picking_strategy == "best":
                winner_idx = int(result["winner_idx"])
            elif args.picking_strategy == "worst":
                winner_idx = int(np.argmin(votes))
            else:  # random
                winner_idx = int(np.random.randint(len(cand_paths)))
            if not (0 <= winner_idx < len(cand_paths)):
                raise RuntimeError(f"winner_idx={winner_idx} out of range")
            with open(cycle_subdir / "ranking.json", "w") as f:
                json.dump({**result, "picking_strategy": args.picking_strategy,
                           "chosen_idx": winner_idx}, f, indent=2)

        # Restore s_t and execute ONLY the winning chunk for real.
        restore_state(s_t)
        reached_success = reached_done = False
        succ_step = None
        n_exec = 0
        print('ACTION SHAPE:', actions_kept[winner_idx, :n_act_exec].shape)
        for a in actions_kept[winner_idx, :n_act_exec]:
            obs, reward, done, info = env.step(a.astype(np.float32))
            obs_hist.append(obs)
            n_exec += 1
            # base_step keeps the downsample phase continuous across cycles within a rollout.
            if (base_step + n_exec) % steps_per_render == 0:
                rollout_rec.write_frame(env.render())
            if float(reward) > 0 or bool(info.get("success", False)):
                reached_success = True
                succ_step = n_exec
            if bool(done):
                reached_done = True
            if reached_success or reached_done:
                break
        return winner_idx, votes, keep, n_exec, reached_success, reached_done, succ_step

    # ---- 5. Rollout loop (single env, reset between rollouts) ----
    rollout_results = []
    global_cycle = 0
    interrupted = False
    try:
        for rollout_idx in range(args.num_rollouts):
            scene_seed = args.env_seed + rollout_idx
            env._seed = scene_seed  # RobomimicImageWrapper.reset() feeds this to the scene RNG
            obs0 = env.reset()
            obs_hist = collections.deque([obs0] * n_obs, maxlen=n_obs)
            if hasattr(policy, "reset"):
                policy.reset()

            r_cand_dir = cand_dir / f"seed{scene_seed}"
            rollout_path = str(media_dir / f"seed{scene_seed}.mp4")
            rollout_rec = VideoRecorder.create_h264(fps=fps, codec="h264", input_pix_fmt="rgb24", crf=crf)
            rollout_rec.start(rollout_path)
            print(colored(f"[rollout {rollout_idx + 1}/{args.num_rollouts}] scene seed {scene_seed}", "cyan"))

            cycle = 0
            total_steps = 0
            success = False
            success_step = None
            cycle_log = []
            stop_reason = "max_steps"
            try:
                while total_steps < max_steps:
                    (winner_idx, votes, keep, n_exec,
                     reached_success, reached_done, succ_step) = run_one_cycle(
                        scene_seed, r_cand_dir, cycle, global_cycle, rollout_rec, total_steps)
                    total_steps += n_exec
                    if reached_success and not success:
                        success = True
                        success_step = total_steps - n_exec + succ_step
                    cycle_log.append({"cycle": cycle, "winner_idx": winner_idx, "votes": votes,
                                      "kept_indices": [int(x) for x in keep]})
                    print(colored(
                        f"  cycle {cycle}: votes={votes} -> {args.picking_strategy} "
                        f"winner_idx={winner_idx} (steps {total_steps}/{max_steps})", "yellow"))
                    cycle += 1
                    global_cycle += 1
                    if success:
                        stop_reason = "success"
                        break
                    if reached_done:
                        stop_reason = "env_done"
                        break
            except KeyboardInterrupt:
                stop_reason = "keyboard_interrupt"
                interrupted = True
            except Exception as e:
                import traceback
                stop_reason = f"exception:{type(e).__name__}"
                print(colored(f"  rollout {rollout_idx} crashed: {e!r}", "red"))
                traceback.print_exc()
            finally:
                rollout_rec.stop()

            rollout_results.append({
                "rollout_idx": rollout_idx,
                "scene_seed": scene_seed,
                "success": bool(success),
                "success_step": success_step,
                "total_steps": total_steps,
                "cycles": cycle,
                "stop_reason": stop_reason,
                "video": rollout_path,
                "per_cycle": cycle_log,
            })
            print(colored(
                f"[rollout {rollout_idx + 1}/{args.num_rollouts}] seed {scene_seed}: "
                f"success={success} (step {success_step}), steps={total_steps}, "
                f"stop_reason={stop_reason}", "green"))
            if interrupted:
                print(colored("Interrupted; stopping remaining rollouts.", "red"))
                break
    finally:
        if server_proc is not None and server_proc.poll() is None:
            print("Shutting down rank server.")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        try:
            env.env.close()
        except Exception as e:
            print(f"env close failed: {e!r}")

    # ---- 6. Finalize ----
    n_done = len(rollout_results)
    n_succ = sum(r["success"] for r in rollout_results)
    success_rate = (n_succ / n_done) if n_done else 0.0
    log = {
        "task": args.task,
        "split": args.split,
        "seed": seed,
        "env_seed_base": args.env_seed,
        "num_rollouts_requested": args.num_rollouts,
        "num_rollouts_completed": n_done,
        "num_success": n_succ,
        "success_rate": success_rate,
        "rollouts": rollout_results,
        "eval_args": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "checkpoint": args.checkpoint,
            "vlm_checkpoint": args.vlm_checkpoint,
            "vlm_task_name": vlm_task_name,
            "picking_strategy": args.picking_strategy,
            "num_samples": args.num_samples,
            "oversample": oversample,
            "n_act_exec": n_act_exec,
            "sampler": args.sampler,
            "num_inference_steps": int(getattr(policy, "num_inference_steps", -1)),
            "pred_horizon": pred_h,
            "n_obs_steps": n_obs,
            "max_steps": max_steps,
            "render_size": args.render_size,
        },
    }
    with open(run_dir / "eval_log.json", "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)

    print(colored(
        f"\nDone. success_rate={success_rate:.3f} ({n_succ}/{n_done} rollouts).", "green"))
    print(colored(f"Log: {run_dir / 'eval_log.json'}", "green"))


if __name__ == "__main__":
    main()
