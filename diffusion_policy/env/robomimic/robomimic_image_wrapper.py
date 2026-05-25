from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite
from robomimic.macros import LANG_EMB_KEY
from robomimic.utils.lang_utils import LangEncoder
from robomimic.utils.obs_utils import process_frame
from robocasa.utils.env_utils import convert_action


class RobomimicImageWrapper(gym.Env):
    def __init__(self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        render_width=None,
        render_height=None,
        render_camera=None,
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        # When render_width/height are set, render() does a fresh high-res frame
        # straight from the MuJoCo sim instead of returning the (low-res) policy
        # observation image. render_camera picks the camera; defaults to the one
        # matching render_obs_key (e.g. robot0_agentview_right_image -> robot0_agentview_right).
        self.render_width = render_width
        self.render_height = render_height
        if render_camera is None and render_obs_key is not None:
            render_camera = render_obs_key[:-len('_image')] if render_obs_key.endswith('_image') else render_obs_key
        self.render_camera = render_camera
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.lang = None
        self.lang_emb = None
        self.lang_encoder = LangEncoder('cpu')
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key == LANG_EMB_KEY:
                min_value, max_value = -100, 100
            elif key.endswith('sin'):
                min_value, max_value = -1, 1
            elif key.endswith('cos'):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def process_obs(self, raw_obs):
        """
        Remaps keys from raw observation to keys expected by diffusion policy
        and performs image processing (normalization + channel reordering)
        """
        obs_mappings = self.shape_meta["obs"]
        policy_obs = {}
        for policy_key, value in obs_mappings.items():
            raw_obs_keys = value.get("lerobot_keys", None)
            if raw_obs_keys is None:
                continue
            raw_obs_key = raw_obs_keys[0]
            policy_obs[policy_key] =  raw_obs[raw_obs_key]
            if value.get("type", "lowdim") == "rgb":
                img = policy_obs[policy_key]
                img_processed = process_frame(img, channel_dim=3, scale=255.)
                policy_obs[policy_key] = img_processed
        return policy_obs

    def get_observation(self, raw_obs=None):
        assert raw_obs is not None, "raw_obs must be provided"
        raw_obs = self.process_obs(raw_obs)
        assert self.lang is not None
        raw_obs[LANG_EMB_KEY] = self.lang_emb
        
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        # if self.init_state is not None:
        #     if not self.has_reset_before:
        #         # the env must be fully reset at least once to ensure correct rendering
        #         self.env.reset()
        #         self.has_reset_before = True

        #     # always reset to the same state
        #     # to be compatible with gym
        #     raw_obs = self.env.reset_to({'states': self.init_state})
        # elif self._seed is not None:
        #     # reset to a specific seed
        #     seed = self._seed
        #     if seed in self.seed_state_map:
        #         # env.reset is expensive, use cache
        #         raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
        #     else:
        #         # robosuite's initializes all use numpy global random state
        #         np.random.seed(seed=seed)
        #         raw_obs = self.env.reset()
        #         state = self.env.get_state()['states']
        #         self.seed_state_map[seed] = state
        #     self._seed = None
        # else:
        #     # random reset
        #     raw_obs = self.env.reset()
        # Per-rollout env seeding: when a seed is set, pass it into the env reset
        # so the kitchen scene (layout/style/object placement) is a deterministic
        # function of this seed. RoboCasaGymEnv.reset(seed=...) reseeds only the
        # env's numpy Generator (self.env.rng) -- it does NOT touch torch, so the
        # policy's denoising noise (torch.randn) still varies run-to-run. Net:
        # same env conditions across runs, distinct outcomes.
        if self._seed is not None:
            raw_obs, info = self.env.reset(seed=self._seed)
        else:
            raw_obs, info = self.env.reset()
        self.lang = raw_obs["annotation.human.task_description"]
        self.lang_emb = self.lang_encoder.get_lang_emb(self.lang).numpy()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        action_converted = convert_action(action)
        raw_obs, reward, done, truncated, info = self.env.step(action_converted)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        # High-res path: render a fresh frame from the MuJoCo sim at the requested
        # resolution. This is independent of the 256x256 policy obs, so the policy
        # input is unchanged -- only the saved video gets the extra detail.
        if self.render_width is not None and self.render_height is not None:
            # self.env.unwrapped is RoboCasaGymEnv; .sim is forwarded to the inner
            # robosuite env. Go through .unwrapped to skip gymnasium's wrapper chain
            # (avoids the "env.sim ... is deprecated" warning).
            img = self.env.unwrapped.sim.render(
                width=self.render_width,
                height=self.render_height,
                camera_name=self.render_camera,
            )
            # MuJoCo's read_pixels returns the frame bottom-up; robosuite's obs
            # pipeline flips it with [::-1], so do the same here to stay upright.
            return np.ascontiguousarray(img[::-1].astype(np.uint8))

        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
