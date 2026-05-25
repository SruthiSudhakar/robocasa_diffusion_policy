"""
Run a trained diffusion policy in RoboCasa simulation, render rollout videos, and save per-task metrics.

This is a thin wrapper around eval_robocasa.eval_task — same defaults, same outputs, just a clearer
file name that matches the project's run_diffusion_policy_*.py convention.

Example:
    python run_diffusion_policy_robocasa.py \
        --checkpoint data/outputs/.../checkpoints/latest.ckpt \
        --output_dir data/evals/atomic_seen_target \
        --task_set atomic_seen \
        --split target \
        --num_rollouts 50 \
        --num_envs 5 \
        --device cuda:0

Outputs land in <output_dir>/<TaskName>/:
    eval_log.json   per-rollout reward, success, video paths
    media/*.mp4     rendered videos for the n_test_vis rollouts (default 4 per task)

To aggregate across tasks into one summary table:
    python diffusion_policy/scripts/get_eval_stats.py --dir <output_dir>
"""
import argparse
import os
import sys

from termcolor import colored
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

from eval_robocasa import eval_task


def main():
    parser = argparse.ArgumentParser(
        description="Run a diffusion policy across a RoboCasa task set and save videos + metrics."
    )
    parser.add_argument("-c", "--checkpoint", required=True,
                        help="Path to the trained .ckpt file.")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Where to write per-task eval_log.json and videos. "
                             "Defaults to <ckpt_dir>/../evals/<ckpt_stem>/<split>/.")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("-t", "--task_set", required=True, nargs="+",
                        help="One or more task-set names from TASK_SET_REGISTRY (e.g. atomic_seen, "
                             "composite_seen) AND/OR individual task names (e.g. CloseToasterOvenDoor) "
                             "to evaluate just specific tasks.")
    parser.add_argument("-n", "--num_rollouts", default=50, type=int,
                        help="Rollouts per task (default: 50).")
    parser.add_argument("-e", "--num_envs", default=14, type=int,
                        help="Parallel envs per task (default: 5). Lower if you're memory-bound.")
    parser.add_argument("-s", "--split", required=True, choices=["pretrain", "target"],
                        help="Kitchen scene/object split. Use 'target' to match the Target (Human) train data.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run a task even if its eval_log.json already exists.")
    parser.add_argument("--sampler", default="ddim", choices=["ddpm", "ddim"],
                        help="Reverse diffusion sampler. 'ddim' + a low --num_inference_steps makes "
                             "rollouts ~10x faster and works on a bs192/DDPM-trained checkpoint without "
                             "retraining. 'ddpm' (default) matches training.")
    parser.add_argument("--num_inference_steps", default=8, type=int,
                        help="Denoising steps at inference (default: the checkpoint's value, 100 for bs192). "
                             "DDIM stays accurate at ~8-10; lowering DDPM this far degrades quality.")
    parser.add_argument("--num_vis", default=-1, type=int,
                        help="Number of test rollouts to render to video (default: the config's value, "
                             "n_test_vis=4). Pass -1 to record every rollout (slower, more disk).")
    parser.add_argument("--max_steps", default=None, type=int,
                        help="Step budget per rollout (default: 1.5x the task horizon, e.g. 675 for "
                             "CloseToasterOvenDoor). Lower it for a tighter budget / faster evals.")
    parser.add_argument("--render_size", default="848x480", type=str,
                        help="Resolution for saved videos as WxH (e.g. 848x480). By default videos "
                             "are the 256x256 policy observation image; this renders a fresh, "
                             "genuinely higher-res frame from the sim instead (policy obs unchanged, "
                             "slightly slower). H264 needs even W and H.")
    parser.add_argument("--render_camera", default=None, type=str,
                        help="Camera name for the high-res video (default: the one matching "
                             "--render_obs_key, e.g. robot0_agentview_right).")
    parser.add_argument("--seed", default=None, type=int,
                        help="Per-run torch seed for the policy's denoising noise. Default: drawn from "
                             "entropy each run, so repeated runs vary (the scenes stay fixed via the env "
                             "seed). Pass a fixed int to reproduce a specific run; the seed used is "
                             "printed and saved to eval_log.json.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # parse --render_size WxH into (width, height) for high-res video rendering.
    render_width = render_height = None
    if args.render_size is not None:
        try:
            render_width, render_height = (int(x) for x in args.render_size.lower().split("x"))
        except ValueError:
            sys.exit(f"--render_size must be WxH (e.g. 848x480), got: {args.render_size}")
        if render_width % 2 or render_height % 2:
            sys.exit(f"--render_size dims must be even for h264, got: {args.render_size}")

    # Each name in --task_set may be either a task-SET name (expands to its tasks)
    # or a single task/env name (e.g. CloseToasterOvenDoor) for targeting one task.
    all_tasks = []
    for name in args.task_set:
        if name in TASK_SET_REGISTRY:
            all_tasks += TASK_SET_REGISTRY[name]
        else:
            all_tasks.append(name)  # treat as an individual task name
    all_tasks = sorted(set(all_tasks))

    print(colored(f"Running eval on {len(all_tasks)} tasks: {all_tasks}", "cyan"))
    print(colored(f"Checkpoint: {args.checkpoint}", "cyan"))
    print(colored(f"Split: {args.split}  |  rollouts/task: {args.num_rollouts}  |  envs: {args.num_envs}",
                  "cyan"))

    for task_i, task in enumerate(all_tasks):
        print(colored(f"[{task_i + 1}/{len(all_tasks)}] running evals for {task}", "yellow"))
        eval_task(
            checkpoint=args.checkpoint,
            base_output_dir=args.output_dir,
            device=args.device,
            task=task,
            num_rollouts=args.num_rollouts,
            num_envs=args.num_envs,
            split=args.split,
            overwrite=args.overwrite,
            sampler=args.sampler,
            num_inference_steps=args.num_inference_steps,
            num_vis=args.num_vis,
            max_steps=args.max_steps,
            seed=args.seed,
            render_width=render_width,
            render_height=render_height,
            render_camera=args.render_camera,
        )

    print(colored("\nDone. Aggregate with:", "green"))
    out_hint = args.output_dir or "<auto-output-dir-printed-above>"
    print(colored(f"  python diffusion_policy/scripts/get_eval_stats.py --dir {out_hint}", "green"))


if __name__ == "__main__":
    main()
