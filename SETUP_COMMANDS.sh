#!/usr/bin/env bash
# RoboCasa atomic-seen 18-task diffusion policy: setup + train + eval commands.
#
# This file is meant to be copy-pasted in chunks across multiple Claude sessions
# (or run section by section). Not designed to run end-to-end as one script —
# Section 5 (full train) is multi-day, Section 6 needs a checkpoint from Section 5.
#
# Sections you can parallelize across Claude instances:
#   - Section 1 install steps are mostly sequential within a single shell
#     (one conda env), so run them in ONE instance.
#   - Section 3 dataset downloads can be split: one Claude per ~6 tasks by
#     splitting the --tasks list.
#   - Section 4 smoke-train, Section 5 full-train, Section 6 eval each run
#     in their own dedicated instance.
#
# ============================================================================
# Section 1 — Install (one-time, ~30 min, run in ONE Claude instance)
# ============================================================================

cd /proj/vondrick3/sruthi/Appaji
conda create -c conda-forge -n robocasa_dp python=3.11 -y
conda activate robocasa_dp

# robosuite master
git clone https://github.com/ARISE-Initiative/robosuite robosuite_new
cd robosuite_new && pip install -e . && cd ..

# robocasa main
git clone https://github.com/robocasa/robocasa robocasa_new
cd robocasa_new && pip install -e .
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets   # ~10 GB
cd ..

# robomimic — MUST be the `robocasa` branch. ARISE master lacks LANG_EMB_KEY,
# LangEncoder, and VisualCoreLanguageConditioned that the fork imports.
git clone -b robocasa https://github.com/ARISE-Initiative/robomimic robomimic_new
cd robomimic_new && pip install -e . && cd ..

# the benchmark diffusion_policy fork (already cloned)
cd robocasa_diffusion_policy
pip install -e .

# --- Dependency stack -------------------------------------------------------
# The fork's conda_environment.yaml is STALE (torch 1.12 / numpy 1.23 /
# diffusers 0.11, inherited from upstream Stanford diffusion_policy). The
# current robocasa 1.0.1 hard-requires numpy==2.2.5 (there is a literal
# `assert numpy.__version__ == "2.2.5"` in robocasa/__init__.py) and mujoco
# 3.3.1. numpy 2 then forces: torch>=2.3 (2.0.1 is built against numpy 1 ABI),
# zarr>=2.18 (older zarr can't import under numpy 2; we keep 2.x because the
# code uses the zarr-2 API: MemoryStore/group/copy_store), diffusers>=0.28
# (0.27 imports the removed huggingface_hub.cached_download), and
# huggingface-hub<1.0 (transformers 4.41.2 / tokenizers 0.19.1 need <1.0).
# The pip "dependency conflict" warnings about robomimic pinning torch==2.0.1
# / numpy==1.23.2 / diffusers==0.11.1 are STALE setup.py pins and are safe to
# ignore — the robocasa branch code runs fine on this stack.
#
# torch first (cu121 wheels work with the A6000 / driver 560 / CUDA 12.6):
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# robocasa canonical pins + numpy-2-compatible zarr/diffusers/hub:
pip install "numpy==2.2.5" "numba==0.61.2" "scipy==1.15.3" "mujoco==3.3.1" \
            "zarr==2.18.3" "numcodecs==0.13.1" "diffusers==0.31.0" \
            "huggingface-hub[cli,hf-transfer]>=0.34.2,<1.0"

# remaining helpers (numpydantic is a lerobot 0.3.3 transitive dep that pip
# misses; accelerate is needed by the workspace import):
pip install hydra-core omegaconf wandb dill av decord termcolor click \
            scikit-video threadpoolctl einops imageio imageio-ffmpeg \
            accelerate numpydantic
cd ..

# Known-good versions confirmed by a passing 2-epoch smoke train (2026-05-23):
#   torch 2.5.1+cu121  torchvision 0.20.1+cu121  numpy 2.2.5  numba 0.61.2
#   scipy 1.15.3  mujoco 3.3.1  zarr 2.18.3  numcodecs 0.13.1  diffusers 0.31.0
#   huggingface-hub 0.36.2  transformers 4.41.2  tokenizers 0.19.1
#   numpydantic 1.8.1  accelerate 1.13.0  lerobot 0.3.3  robocasa 1.0.1
#   robomimic 0.3.0 (robocasa branch)  pandas 3.0.3  tianshou 0.4.10
cd robocasa_diffusion_policy

# ============================================================================
# Section 2 — Verify install (must all print OK before continuing)
# ============================================================================

python -c "
import gymnasium as gym, robocasa
from robocasa.utils.dataset_registry import TARGET_TASKS, DATASET_SOUP_REGISTRY, TASK_SET_REGISTRY
print('atomic_seen tasks:', TARGET_TASKS['atomic_seen'])
print('soup exists:', 'target_atomic_seen' in DATASET_SOUP_REGISTRY)
print('task_set exists:', 'atomic_seen' in TASK_SET_REGISTRY)
env = gym.make('robocasa/PickPlaceCounterToCabinet', split='target', seed=0); env.reset()
print('OK env constructed')
"


# ============================================================================
# Section 3 — Download Target (Human) data for the 18 atomic-seen tasks
# (~200-400 GB. Can be split across multiple Claude instances by partitioning
#  the --tasks list.)
# ============================================================================

# Confirm flag names first (they vary slightly between robocasa releases):
python -m robocasa.scripts.download_datasets --help

# # All 18 at once (one Claude instance):
# python -m robocasa.scripts.download_datasets \
#     --split target --source human \
#     --tasks CloseBlenderLid CloseFridge CloseToasterOvenDoor CoffeeSetupMug \
#             NavigateKitchen OpenCabinet OpenDrawer OpenStandMixerHead \
#             PickPlaceCounterToCabinet PickPlaceCounterToStove \
#             PickPlaceDrawerToCounter PickPlaceSinkToCounter \
#             PickPlaceToasterToCounter SlideDishwasherRack \
#             TurnOffStove TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet

# --- OR ---
# Parallelize across THREE Claude instances (each downloads ~6 tasks).
# Make sure all three instances have the conda env active.
#
Done:
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks NavigateKitchen
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks CloseBlenderLid CloseFridge CloseToasterOvenDoor
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks CoffeeSetupMug NavigateKitchen OpenCabinet
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks PickPlaceToasterToCounter SlideDishwasherRack TurnOffStove
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks OpenDrawer OpenStandMixerHead PickPlaceCounterToCabinet
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks PickPlaceCounterToStove PickPlaceDrawerToCounter PickPlaceSinkToCounter
python -m robocasa.scripts.download_datasets --split target --source human \
    --tasks TurnOnElectricKettle TurnOnMicrowave TurnOnSinkFaucet

# Verify every soup path resolves to a real file.
# NOTE: the current robocasa renamed the soup from `posttrain_atomic_seen`
# to `target_atomic_seen`. The fork's yaml has been patched to match.
python -c "
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY
from robocasa.macros import DATASET_BASE_PATH
import os, robocasa
base = DATASET_BASE_PATH or os.path.join(os.path.dirname(os.path.dirname(robocasa.__file__)), 'datasets')
for item in DATASET_SOUP_REGISTRY['target_atomic_seen']:
    p = item['path']
    if not os.path.isabs(p): p = os.path.join(base, p)
    print(('OK ' if os.path.exists(p) else 'MISSING '), p)
"


# ============================================================================
# Section 4 — Smoke train (~5 min, confirms pipeline before committing to a
# multi-day run)
# ============================================================================

cd /proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy

HYDRA_FULL_ERROR=1 python train.py \
    --config-name=train_diffusion_transformer_bs192 \
    task=robocasa/target_atomic_seen \
    training.device=cuda:0 \
    training.num_epochs=2 \
    logging.mode=offline


# ============================================================================
# Section 5 — Full train (multi-day on a single A100; dedicated Claude instance)
# ============================================================================
Convert data:
python scripts/reencode_allintra.py --dataset-paths

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
OPENCV_FFMPEG_CAPTURE_OPTIONS="threads;1" \
HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
    train.py \
    --config-name=train_diffusion_unet_hybrid_workspace \
    task=robocasa/target_atomic_seen \
    task.dataset.dataset_soup=null \
    '+task.dataset.dataset_paths=[/proj/vondrick3/sruthi/Appaji/robocasa_new/datasets/v1.0/target/atomic/CloseToasterOvenDoor/20250818/lerobot]' \
    'hydra.run.dir=data/jgd/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_jgd_CloseToasterOvenDoor' \
    training.checkpoint_every=20 \
    training.val_every=20 \
    training.sample_every=20

# ============================================================================
# Section 6 — Evaluate + render (needs a trained checkpoint from Section 5)
# ============================================================================

cd /proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy
conda activate robocasa_dp

CKPT=/proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/data/jgd/2026.05.26/10.38.08_train_diffusion_unet_hybrid_target_atomic_seen_jgd_PickPlaceToasterOvenToCounter/checkpoints/epoch=0075-train_loss=0.0021.ckpt
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python run_diffusion_policy_robocasa.py \
    --checkpoint "$CKPT" --task_set PickPlaceToasterOvenToCounter --split pretrain --render_camera robot0_eye_in_hand

CKPT=/proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/released_checkpoints/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=1400-test_mean_score=-1.000.ckpt
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python run_diffusion_policy_robocasa.py \
--checkpoint "$CKPT" --task_set TurnOnMicrowave --split target

# ============================================================================
# Section 7 — Evaluate + render with VLM ranking (needs a trained checkpoint from Section 5)
# ============================================================================
cd /proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy
conda activate robocasa_dp

RANK_DIR=/proj/vondrick3/sruthi/Appaji/rc_rank/PickPlaceCounterToDrawer
VLM_CKPT=/proj/vondrick3/sruthi/Appaji/trl/outputs/PickPlaceCounterToDrawer_20260526_213216_robocasa/checkpoint-1000
CKPT=/proj/vondrick3/sruthi/Appaji/robocasa_diffusion_policy/released_checkpoints/diffusion_policy/17.40.09_train_diffusion_transformer_hybrid_pretrain_human300/checkpoints/epoch=1400-test_mean_score=-1.000.ckpt
TASK=PickPlaceCounterToDrawer
SPLIT=pretrain

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python run_diffusion_policy_robocasa_vlm_ranking.py \
--checkpoint "$CKPT" \
--task "$TASK" \
--split "$SPLIT" \
--num-samples 1


CUDA_VISIBLE_DEVICES=1 /proj/vondrick3/sruthi/miniconda3/envs/vlmoverlay/bin/python \
    /proj/vondrick3/sruthi/Appaji/trl/examples/scripts/myscripts/rank_serve_robocasa.py \
    --inbox_dir $RANK_DIR/inbox --done_dir $RANK_DIR/done --error_dir $RANK_DIR/error \
    --checkpoint "$VLM_CKPT" \
    --task_name "$TASK" \
    --gpu_ids 0 --batch_size 10 --max_pixels 960x540

CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python run_diffusion_policy_robocasa_vlm_ranking.py \
--checkpoint "$CKPT" \
--task "$TASK" \
--split "$SPLIT" \
--no-launch-rank-server \
--rank-dir $RANK_DIR \
--vlm-checkpoint $VLM_CKPT
