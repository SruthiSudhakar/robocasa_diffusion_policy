import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from omegaconf import OmegaConf
from functools import partial
from pathlib import Path

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import gymnasium as gym

import robocasa
import robocasa.utils.lerobot_utils as LU

# these lines necessary to prevent all the CPU cores being taken up
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def create_env(split, env_name, seed=None):
    env = gym.make(
        f"robocasa/{env_name}",
        split=split,
        seed=seed
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            render_width=None,
            render_height=None,
            render_camera=None,
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            env_kwargs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # disable object state observation
        self.env_kwargs = OmegaConf.to_container(env_kwargs) if env_kwargs is not None else {}
        env_name = self.env_kwargs["env_name"]

        rotation_transformer = None
        if abs_action:
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn(env_i):
            # set seed for env appropriately
            if "seed" in self.env_kwargs:
                self.env_kwargs["seed"] += env_i
            robocasa_env = create_env(
                split=self.env_kwargs["split"], 
                env_name=self.env_kwargs["env_name"],
                seed=self.env_kwargs.get("seed", None)
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            # robocasa_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robocasa_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        render_width=render_width,
                        render_height=render_height,
                        render_camera=render_camera
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robocasa_env = create_env(
                split=self.env_kwargs["split"], 
                env_name=env_name,
                seed=self.env_kwargs.get("seed", None)
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robocasa_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        render_width=render_width,
                        render_height=render_height,
                        render_camera=render_camera
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [partial(env_fn, env_i) for env_i in range(n_envs)]
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        assert n_train == 0, "Training envs not supported"
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    # name the video by its (deterministic) env seed instead of a random
                    # id, so the same rollout lines up by filename across runs with the
                    # same variables (e.g. media/seed10000.mp4) and is easy to diff.
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f"seed{seed}.mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                # Pin this rollout's kitchen scene to its (deterministic) seed.
                # RobomimicImageWrapper.reset() passes this into env.reset(seed=...),
                # which reseeds only the env's numpy scene RNG -- the policy's torch
                # noise is untouched, so the same scene plays out with distinct
                # actions across runs. seed = test_start_seed + rollout_index, so
                # every rollout gets a distinct, reproducible scene.
                env.env.env._seed = seed

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # Padding init fn for partial last chunks. AsyncVectorEnv needs exactly
        # n_envs init fns, so a short final chunk is padded out to n_envs (see
        # run()). The padding envs run a throwaway rollout whose results are
        # sliced away -- but they must NOT write a video. Previously padding
        # reused env_init_fn_dills[0], whose init_fn points the recorder at
        # media/seed{test_start_seed}.mp4 with rendering enabled; when a chunk
        # was short by >=2, multiple worker processes wrote that same file
        # concurrently and corrupted it (always seed{test_start_seed}, in every
        # rollout). Disable rendering here so file_path stays None.
        def pad_init_fn(env, seed=test_start_seed):
            assert isinstance(env.env, VideoRecordingWrapper)
            env.env.video_recoder.stop()
            env.env.file_path = None
            assert isinstance(env.env.env, RobomimicImageWrapper)
            env.env.env.init_state = None
            env.env.env._seed = seed
        self.pad_init_fn_dill = dill.dumps(pad_init_fn)

        env = AsyncVectorEnv(
            env_fns,
            dummy_env_fn=dummy_env_fn,
            # RoboCasa's observation space is a custom gym.Space, which the
            # shared-memory path can't batch. Use pipe-based transfer instead.
            shared_memory=False,
        )
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.robosuite_fps = robosuite_fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        # env step at which each rollout first succeeded (None if it never did)
        all_success_steps = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                # Pad with a non-rendering init fn so padding envs never write a
                # (duplicate, concurrently-corrupted) video. See pad_init_fn.
                this_init_fns.extend([self.pad_init_fn_dill]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_kwargs['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                ac_size = list(env_action.shape)
                ac_size[-1] = 1
                # base_ac = np.array([0, 0, 0, 0, -1])

                # # Expand base_ac to shape (1, 8, 5) so it can be concatenated along the last axis
                # base_ac_expanded = np.tile(base_ac, ac_size)

                # # Concatenate along the last axis (axis=-1)
                # env_action = np.concatenate([env_action, base_ac_expanded], axis=-1)
                obs, reward, done, info = env.step(env_action)
                # done = np.all(done)
                # for robocasa switch to the proper success check
                done = np.all(done) or np.all([this_info["success"][0] for this_info in info])
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_success_steps[this_global_slice] = env.call('get_attr', 'success_step')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        success_rate = sum([np.max(all_rewards[i]) > 0 for i in range(n_inits)]) / n_inits
        print(f"Success rate: {success_rate}")
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # step (and wall-clock time) at which this rollout first succeeded.
            # None means the task was never completed within the step budget.
            success_step = all_success_steps[i]
            log_data[prefix+f'success_step_{seed}'] = success_step
            log_data[prefix+f'success_time_sec_{seed}'] = (
                None if success_step is None else success_step / self.robosuite_fps)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        env_name = self.env_kwargs["env_name"]
        log_data[f'success_rate/{env_name}'] = success_rate
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        ### WARNING: doesn't work with anything other than arm+gripper action! ###

        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def close(self):
        if not isinstance(self.env, SyncVectorEnv):
            return
        
        # only for SyncVectorEnv
        env_list = self.env.envs
        for env in env_list:
            env_chain = [env]
            while True:
                if hasattr(env, "env"):
                    env = env.env
                else:
                    break
                env_chain = [env] + env_chain
            
            for env in env_chain:
                if hasattr(env, "close"):
                    env.close()
                del env
