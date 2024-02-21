import argparse
import itertools
import logging
import os
import shutil

import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.AutoencoderKL import AutoencoderKL
from models.inversion_adapter import InversionAdapter
from utils.encode_text_word_embedding import encode_text_word_embedding
from utils.image_from_pipe import generate_images_from_tryon_pipe
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

torch.multiprocessing.set_sharing_strategy('file_system')\

outputlist = ['image', 'pose_map', 'captions', 'inpaint_mask', 'im_mask', 'category', 'im_name']
train_dataset = DressCodeDataset(
    dataroot_path='data/noun_chunks/DressCode',
    phase='train',
    order='paired',
    radius=5,
    category=['lower_body'],
    size=(512, 384),
    outputlist=tuple(outputlist)
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=16,
    num_workers=8,
)