"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""
import gymnasium as gym
import robocasa
from diffusion_policy.common.pytorch_util import dict_apply

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from omegaconf import OmegaConf
import copy

import os
import pathlib
import argparse
import datetime
import hydra
import torch
import dill
import wandb
import json
from termcolor import colored
from diffusion_policy.workspace.base_workspace import BaseWorkspace

from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY


def swap_to_ddim(policy, num_inference_steps=None):
    """Swap a (DDPM-trained) policy's reverse sampler to DDIM for fast eval.

    Training only depends on the forward noising process + epsilon prediction,
    which DDPM and DDIM share, so a DDPM-trained checkpoint can be sampled with
    DDIM without retraining. Mirrors the training betas exactly and adds DDIM's
    set_alpha_to_one / steps_offset (matching the *_ddim.yaml configs).
    """
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    base = policy.noise_scheduler.config
    policy.noise_scheduler = DDIMScheduler(
        num_train_timesteps=base.num_train_timesteps,
        beta_start=base.beta_start,
        beta_end=base.beta_end,
        beta_schedule=base.beta_schedule,
        prediction_type=base.prediction_type,
        clip_sample=base.clip_sample,
        set_alpha_to_one=True,
        steps_offset=0,
    )
    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps
    print(colored(
        f"Using DDIM sampler with num_inference_steps={policy.num_inference_steps}", "cyan"))


def eval_task(checkpoint, base_output_dir, device, task, num_rollouts, num_envs, split, overwrite,
              sampler="ddpm", num_inference_steps=None, num_vis=None, max_steps=None, seed=None,
              render_width=None, render_height=None, render_camera=None):
    if base_output_dir is None:
        base_output_dir = os.path.join(os.path.dirname(checkpoint), "../evals", os.path.basename(checkpoint).replace(".ckpt", ""), split)

    # resolve the per-run noise seed up front so it can go in the output dir name.
    # (it's only *applied* to torch later, after the workspace reseeds the global RNGs.)
    if seed is None:
        seed = torch.seed() & 0x7FFFFFFF  # draw from entropy; keep it printable/reproducible
    seed = int(seed)

    # stamp each run with a datetime + seed so repeated runs land in their own
    # <task>_<num_envs>_<num_rollouts>_<datetime>_seed<seed> dir instead of being
    # skipped as already-existing.
    run_stamp = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    output_dir = os.path.join(base_output_dir, f"{task}_{num_envs}_{num_rollouts}_{run_stamp}_seed{seed}")

    out_path = os.path.join(output_dir, 'eval_log.json')
    if overwrite is False and os.path.exists(out_path):
        # click.confirm(f"Output path {out_path} already exists! Overwrite?", abort=True)
        print(f"Eval stats path {out_path} already exists! Skipping.")
        return

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = copy.deepcopy(OmegaConf.to_container(cfg))
    cfg["task"]["env_runner"]["env_kwargs"] = {
        "split": split,
        "seed": 1111111,
        "env_name": task,
    }
    cfg = OmegaConf.create(cfg)

    horizon = get_task_horizon(task=task)
    
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = num_rollouts
    # number of test rollouts to render to video (config default is n_test_vis=4).
    # only the first n_test_vis envs enable rendering; pass -1 to record every rollout.
    if num_vis is not None:
        cfg.task.env_runner.n_train_vis = 0
        cfg.task.env_runner.n_test_vis = num_rollouts if num_vis < 0 else num_vis

    # set dataset path and horizon
    # step budget per rollout: default is 1.5x the task horizon; override with --max_steps.
    cfg.task.env_runner.max_steps = int(max_steps) if max_steps is not None else int(horizon * 1.5)
    cfg.task.env_runner.n_envs = num_envs

    # optional high-res video rendering: render saved mp4s from the sim at this
    # resolution (e.g. 848x480) instead of using the 256x256 policy obs image.
    # leaves the policy's observations untouched.
    if render_width is not None and render_height is not None:
        cfg.task.env_runner.render_width = int(render_width)
        cfg.task.env_runner.render_height = int(render_height)
        if render_camera is not None:
            cfg.task.env_runner.render_camera = render_camera

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # optionally swap the reverse sampler to DDIM for fast eval (no retraining)
    if sampler == "ddim":
        swap_to_ddim(policy, num_inference_steps=num_inference_steps)
    elif num_inference_steps is not None:
        # keep DDPM but change step count (note: DDPM degrades at low steps)
        policy.num_inference_steps = num_inference_steps

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # The workspace constructor reseeds the global RNGs (torch/numpy/random) to the
    # fixed cfg.training.seed, so every run draws the *same* denoising noise and the
    # eval is bit-identical across runs. Reseed ONLY torch here to vary the policy's
    # sampling noise (torch.randn in conditional_sample) while leaving numpy/random
    # untouched -- the env scenes are numpy-driven off the explicit env seed above,
    # so they stay identical. Pass --seed to reproduce a specific run (resolved above).
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(colored(f"Per-run noise seed (torch only): {seed}", "cyan"))

    # run eval
    try_num = 1
    MAX_TRIES = 5
    while try_num <= MAX_TRIES:
        env_runner = None
        runner_log = None
        # try:
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)
        runner_log = env_runner.run(policy)
        # except Exception as e:
        #     print(f"Excpetion in env_runner (try {try_num})")
        #     print(e)
        #     print()
        #     try_num += 1
        #     continue
        
        break
    
    # dump log to json
    if runner_log is not None:
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = value
        # record exactly what produced this rollout so it can be reproduced later.
        # both the raw args and the *resolved* values (after defaults kick in).
        json_log['eval_args'] = {
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'command': ' '.join(sys.argv),
            'checkpoint': checkpoint,
            'task': task,
            'split': split,
            'num_rollouts': num_rollouts,
            'num_envs': num_envs,
            'sampler': sampler,
            'num_vis': num_vis,
            'seed': int(seed),
            # resolved values actually used (defaults applied):
            'max_steps': int(cfg.task.env_runner.max_steps),
            'max_steps_was_default': max_steps is None,
            'num_inference_steps': int(getattr(policy, 'num_inference_steps', -1)),
            'n_action_steps': int(cfg.task.env_runner.n_action_steps),
            'n_obs_steps': int(cfg.task.env_runner.n_obs_steps),
        }
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # close and delete everything
    if env_runner is not None:
        env_runner.close()
    del policy
    del workspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-o', '--output_dir', default=None)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-t', '--task_set', required=True, nargs='+')
    parser.add_argument('-n', '--num_rollouts', default=50, type=int)
    parser.add_argument('-e', '--num_envs', default=14, type=int)
    parser.add_argument('-s', '--split', required=True)
    parser.add_argument('--sampler', default='ddpm', choices=['ddpm', 'ddim'],
                        help="Reverse sampler. 'ddim' is fast (use with low --num_inference_steps); "
                             "works on a DDPM-trained checkpoint without retraining.")
    parser.add_argument('--num_inference_steps', default=None, type=int,
                        help="Denoising steps at inference. Lower = faster. DDIM stays accurate at ~8-10; "
                             "DDPM needs ~50-100.")
    parser.add_argument('--num_vis', default=None, type=int,
                        help="Number of test rollouts to render to video. Default keeps the checkpoint "
                             "config's value (n_test_vis=4). Pass -1 to record every rollout (slower, "
                             "more GPU memory).")
    parser.add_argument('--seed', default=None, type=int,
                        help="Per-run torch seed for the policy's denoising noise. Default: drawn from "
                             "entropy each run (so repeated runs vary). Pass a fixed int to reproduce a "
                             "specific run. Does NOT change env scenes (those use the fixed env seed).")
    args = parser.parse_args()

    all_tasks = []
    for task_soup_i in args.task_set:
        all_tasks += TASK_SET_REGISTRY[task_soup_i]
    all_tasks = set(all_tasks)

    for task_i, task in enumerate(all_tasks):
        print(colored(f"[{task_i+1}/{len(all_tasks)}] running evals for {task}", "yellow"))
        eval_task(args.checkpoint, args.output_dir, args.device, task, args.num_rollouts, args.num_envs,
                  args.split, overwrite=False,
                  sampler=args.sampler, num_inference_steps=args.num_inference_steps,
                  num_vis=args.num_vis, seed=args.seed)

if __name__ == '__main__':
    main()