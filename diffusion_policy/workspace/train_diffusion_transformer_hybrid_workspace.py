if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import math
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import pickle
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import broadcast_object_list
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import time

# hide wandb warnings
import logging
logging.getLogger("wandb").setLevel(logging.ERROR)


OmegaConf.register_new_resolver("eval", eval, replace=True)


class InfiniteRandomSampler(torch.utils.data.Sampler):
    """Yields shuffled dataset indices forever, reshuffling on every pass.

    Used so the training DataLoader's iterator is created exactly once and never
    raises StopIteration. With a fixed number of steps per "epoch"
    (max_train_steps) that exceeds the number of batches in the data, the naive
    approach recreates the iterator whenever it exhausts. Even with
    persistent_workers=True (workers are not respawned) that recreation drains
    every worker's prefetch queue, so the next batch only arrives once the cold
    pipeline refills -- a multi-minute stall, ~9x per epoch here. An infinite
    sampler keeps the workers prefetching continuously instead.

    The order is deterministic given `seed` and identical across distributed
    processes, which Accelerate's split_batches=True requires: every process
    slices the same global batch, so they must agree on index order.
    """

    def __init__(self, data_source, seed=0):
        self.num_samples = len(data_source)
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        pass_idx = 0
        while True:
            g.manual_seed(self.seed + pass_idx)
            yield from torch.randperm(self.num_samples, generator=g).tolist()
            pass_idx += 1

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        timeout_handler = InitProcessGroupKwargs(timeout=timedelta(hours=3)) # have processes wait up to 3 hours for evals to finish
        cfg = copy.deepcopy(self.cfg)
        # set split_batches=True so that effective batch size stays the same regardless of num GPUs
        accelerator = Accelerator(log_with='wandb', kwargs_handlers=[ddp_kwargs, timeout_handler], split_batches=True)

        if cfg.training.debug:
            cfg.logging.project = "debug"

        if accelerator.is_main_process:
            data_to_share = self.output_dir
        else:
            data_to_share = None
        data_list = [data_to_share]
        broadcast_object_list(data_list)
        self._output_dir = data_list[0]

        # resume training
        # Decide whether we are actually resuming from a checkpoint *before*
        # starting the wandb run. Otherwise wandb resumes the previous run (and
        # its step counter) while global_step resets to 0 because no checkpoint
        # had been saved yet -> every log lands below the current wandb step and
        # is silently dropped.
        resumed_from_ckpt = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                resumed_from_ckpt = True

        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        # Only resume the wandb run if we actually restored a checkpoint, so the
        # wandb step pointer and global_step stay in sync.
        wandb_cfg['resume'] = 'allow' if resumed_from_ckpt else 'never'

        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )
        
        if "ckpt_path" in cfg.task and cfg.task.ckpt_path is not None:
            accelerator.print(f"Initializing from checkpoint {cfg.task.ckpt_path}")
            if cfg.training.resume:
                self.load_checkpoint(path=cfg.task.ckpt_path)
            else:
                self.load_checkpoint(path=cfg.task.ckpt_path, include_keys=[])

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        dataloader_cfg = copy.deepcopy(OmegaConf.to_container(cfg.dataloader))
        custom_sampler = None
        if hasattr(dataset, "get_dataset_sampler"):
            custom_sampler = dataset.get_dataset_sampler()
        if custom_sampler is not None:
            dataloader_cfg["sampler"] = custom_sampler
            dataloader_cfg["shuffle"] = False
            print("using custom dataloader sampler")
        else:
            # Infinite sampler so the train iterator is created once and never
            # exhausts -> workers keep prefetching, no per-epoch pipeline stall.
            # max_train_steps (not the dataset size) defines the epoch length.
            dataloader_cfg["sampler"] = InfiniteRandomSampler(
                dataset, seed=cfg.training.seed)
            dataloader_cfg["shuffle"] = False
            # The infinite sampler has no len(), so the training loop drives epoch
            # length from max_train_steps. If unset, default it to one full pass
            # over the dataset (preserves the original "len(dataloader)" semantics).
            if cfg.training.max_train_steps is None:
                batch_size = dataloader_cfg["batch_size"]
                cfg.training.max_train_steps = math.ceil(len(dataset) / batch_size)
        train_dataloader = DataLoader(dataset, **dataloader_cfg)
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))
        
        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        if cfg.training.max_train_steps is None:
            num_training_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        else:
            num_training_steps = (cfg.training.max_train_steps * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        ### Soroush: this part is removed, created dynamically when needed and closed
        # # configure env
        # if cfg.training.rollout_every is not None:
        #     env_runner: BaseImageRunner
        #     env_runner = hydra.utils.instantiate(
        #         cfg.task.env_runner,
        #         output_dir=self.output_dir)
        #     assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        # Created once for all of training; the infinite sampler means this is
        # never re-iter()'d, so the worker prefetch pipeline never goes cold.
        train_dataloader_iter = iter(train_dataloader)
        print("steps per epoch (max_train_steps):", cfg.training.max_train_steps)
        print("dataset length:", len(dataset))
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

            cfg.task.env_runner.n_test = 5
            cfg.task.env_runner.max_steps = 40

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                # max_train_steps = cfg.training.max_train_steps
                # tqdm_kwargs = {}
                # if max_train_steps is not None:
                #     tqdm_kwargs["total"] = max_train_steps
                # with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                #         leave=False, mininterval=cfg.training.tqdm_interval_sec, **tqdm_kwargs) as tepoch:
                #     for batch_idx, batch in enumerate(tepoch):

                assert cfg.training.max_train_steps is not None
                for batch_idx in tqdm.tqdm(range(cfg.training.max_train_steps), desc=f"Training epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec):
                    # Infinite sampler -> never StopIteration; no iterator reset.
                    batch = next(train_dataloader_iter)
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    raw_loss = self.model(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(accelerator.unwrap_model(self.model))

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    # tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    is_last_batch = (batch_idx == (cfg.training.max_train_steps-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        accelerator.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                    # if (cfg.training.max_train_steps is not None) \
                    #     and batch_idx >= (cfg.training.max_train_steps-1):
                    #     break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if cfg.training.rollout_every is not None:
                    if self.epoch > 0 and (self.epoch % cfg.training.rollout_every) == 0:
                        if accelerator.is_main_process:
                            env_runner: BaseImageRunner
                            env_runner = hydra.utils.instantiate(
                                cfg.task.env_runner,
                                output_dir=self.output_dir)
                            assert isinstance(env_runner, BaseImageRunner)

                            runner_log = env_runner.run(policy)
                            # log all
                            step_log.update(runner_log)

                            # close and discard env_runner
                            env_runner.close()
                            del env_runner
                        
                        accelerator.wait_for_everyone()

                # run validation on the held-out trajectories (val_ratio split)
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0:
                    # DDP wrapper doesn't expose compute_loss; use the unwrapped module
                    val_model = accelerator.unwrap_model(self.model)
                    was_training = val_model.training
                    val_model.eval()
                    with torch.no_grad():
                        val_losses = list()
                        for batch_idx, batch in enumerate(tqdm.tqdm(val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec)):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss = val_model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.stack(val_losses))
                            # average across processes so all ranks log the same value
                            val_loss = accelerator.reduce(val_loss, reduction="mean")
                            step_log['val_loss'] = val_loss.item()
                    if was_training:
                        val_model.train()

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    if accelerator.is_main_process:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']
                            
                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse
                    
                    accelerator.wait_for_everyone()
                
                # checkpoint
                if self.epoch > 0 and (self.epoch % cfg.training.checkpoint_every) == 0:
                    if accelerator.is_main_process:
                        model_ddp = self.model
                        self.model = accelerator.unwrap_model(self.model)
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)

                        self.model = model_ddp
                    
                    accelerator.wait_for_everyone()
                # ========= eval end for this epoch ==========

                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
