import os, sys
import numpy as np
import torch
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
from config import *
import wandb
import datetime
import argparse

from torch_compat import load_full

from config import Config_Generative_Model
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM_eval

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')

def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

def wandb_init(config):
    wandb.init( project="dreamdiffusion",
                group='eval',
                anonymous="allow",
                config=config,
                reinit=True)

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--root', type=str, default='../dreamdiffusion/')
    parser.add_argument('--dataset', type=str, default='GOD')
    parser.add_argument('--model_path', type=str)

    parser.add_argument('--splits_path', type=str, default=None,
                        help='Path to dataset splits.')
    parser.add_argument('--eeg_signals_path', type=str, default=None,
                        help='Path to EEG signals data.')

    parser.add_argument('--config_patch', type=str, default=None,
                        help='sd config path.')
    
    parser.add_argument('--imagenet_path', type=str, default=None,
                        help='imagenet path.')

    parser.add_argument('--encoder_path', type=str, default=None,
                        help='Path to text-align fine-tuned encoder checkpoint. '
                             'If provided, replaces the EEG encoder in the base eLDM model.')

    return parser


if __name__ == '__main__':
    # Load default config
    default_config = Config_Generative_Model()

    args = get_args_parser()
    args = args.parse_args()

    # Use config defaults if args not provided
    if args.model_path is None:
        args.model_path = default_config.model_path
    if args.splits_path is None:
        args.splits_path = default_config.splits_path
    if args.eeg_signals_path is None:
        args.eeg_signals_path = default_config.eeg_signals_path
    if args.config_patch is None:
        args.config_patch = default_config.config_patch
    if args.imagenet_path is None:
        args.imagenet_path = default_config.imagenet_path
    if args.encoder_path is None:
        args.encoder_path = default_config.encoder_path

    root = args.root
    target = args.dataset

    # Print loaded configuration
    print("=" * 60)
    print("Configuration Loaded:")
    print("=" * 60)
    print(f"  root            : {root}")
    print(f"  dataset         : {target}")
    print(f"  model_path      : {args.model_path}")
    print(f"  encoder_path    : {args.encoder_path}")
    print(f"  eeg_signals_path: {args.eeg_signals_path}")
    print(f"  splits_path     : {args.splits_path}")
    print(f"  config_patch    : {args.config_patch}")
    print(f"  imagenet_path   : {args.imagenet_path}")
    print("=" * 60)

    sd = load_full(args.model_path, map_location='cpu')
    config = sd['config']
    # update paths
    config.root_path = root

    # Override checkpoint snapshot (sd['config'] ignores edits to config.py)
    config.num_samples = 5
    config.ddim_steps = 250

    output_path = os.path.join(config.root_path, 'results', 'eval',  
                    '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        transforms.Resize((512, 512)),
        # random_crop(config.img_size-crop_pix, p=0.5),
        # transforms.Resize((256, 256)), 
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)), 
        channel_last
    ])

    
    dataset_train, dataset_test = create_EEG_dataset(eeg_signals_path = args.eeg_signals_path, 
                splits_path = args.splits_path, imagenet_path=args.imagenet_path,
                image_transform=[img_transform_train, img_transform_test], subject = 4)
    num_voxels = dataset_test.dataset.data_len



    # create generateive model
    generative_model = eLDM_eval(args.config_patch, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond)
    # m, u = model.load_state_dict(pl_sd, strict=False)
    generative_model.model.load_state_dict(sd['model_state_dict'], strict=False)
    print('load ldm successfully')

    if args.encoder_path is not None:
        enc_sd = load_full(args.encoder_path, map_location='cpu')
        enc_key = "model" if "model" in enc_sd else "model_state_dict"
        enc_weights = enc_sd[enc_key]
        prefixed = {"cond_stage_model.mae." + k: v for k, v in enc_weights.items()}
        m, u = generative_model.model.load_state_dict(prefixed, strict=False)
        print(f"Injected fine-tuned encoder from {args.encoder_path}")
        print(f"  replaced {len(prefixed) - len(u)} keys, {len(u)} unexpected")

    state = sd['state']
    os.makedirs(output_path, exist_ok=True)
    grid, _ = generative_model.generate(dataset_train, config.num_samples, 
                config.ddim_steps, config.HW, 5) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    
    grid_imgs.save(os.path.join(output_path, f'./samples_train.png'))

    grid, samples = generative_model.generate(dataset_test, config.num_samples, 
                config.ddim_steps, config.HW, limit=5, state=state, output_path = output_path) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))


    grid_imgs.save(os.path.join(output_path, f'./samples_test.png'))
