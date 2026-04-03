"""
Stage C - Generate images from EEG signals using the eLDM pipeline.
Saves per-sample folders with GT image, generated images, and text caption.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange, repeat
from PIL import Image

from torch_compat import load_full
from config import Config_Generation
from dataset import create_EEG_dataset
from dc_ldm.ldm_for_eeg import eLDM_eval
from dc_ldm.models.diffusion.plms import PLMSSampler


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0
    return img


def caption_path(text_root, image_name):
    wnid = image_name.split("_")[0]
    return os.path.join(text_root, wnid, f"{image_name}_caption.txt")


def get_image_name(splitter, idx):
    """Retrieve image_name for the idx-th sample in a Splitter."""
    ds = splitter.dataset
    raw_idx = splitter.split_idx[idx]
    image_idx = int(ds.data[raw_idx]["image"])
    return ds.images[image_idx]


def get_gt_image_path(imagenet_path, image_name):
    wnid = image_name.split("_")[0]
    return os.path.join(imagenet_path, wnid, image_name + ".JPEG")


def parse_args():
    parser = argparse.ArgumentParser("Stage C: EEG -> Images (CLI overrides Config_Generation)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--output_base", type=str, default=None)
    parser.add_argument("--text_data", type=str, default=None)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--eeg_signals_path", type=str, default=None)
    parser.add_argument("--config_patch", type=str, default=None)
    parser.add_argument("--imagenet_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--batch_size_accel", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    config = Config_Generation()
    args = parse_args()

    for key in ["model_path", "encoder_path", "output_base", "text_data",
                "splits_path", "eeg_signals_path", "config_patch", "imagenet_path",
                "num_samples", "ddim_steps", "batch_size_accel"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    if config.inject_encoder and config.encoder_path is not None:
        model_name = os.path.splitext(os.path.basename(config.encoder_path))[0]
    else:
        model_name = "baseline"
    output_dir = os.path.join(config.output_base, model_name)

    print("=" * 60)
    print("Stage C: EEG -> Images")
    print("=" * 60)
    print(f"  model_path      : {config.model_path}")
    print(f"  inject_encoder  : {config.inject_encoder}")
    print(f"  encoder_path    : {config.encoder_path if config.inject_encoder else 'N/A (baseline)'}")
    print(f"  output_dir      : {output_dir}")
    print(f"  text_data       : {config.text_data}")
    print(f"  num_samples     : {config.num_samples}")
    print(f"  ddim_steps      : {config.ddim_steps}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load eLDM checkpoint ---
    sd = load_full(config.model_path, map_location="cpu")
    ckpt_config = sd["config"]
    ckpt_config.root_path = config.root_path

    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((512, 512)), channel_last
    ])
    _, dataset_test = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        imagenet_path=config.imagenet_path,
        image_transform=[img_transform_test, img_transform_test],
        subject=config.subject,
    )
    num_voxels = dataset_test.dataset.data_len
    print(f"Test samples: {len(dataset_test)}")

    # --- build generative model ---
    generative_model = eLDM_eval(
        config.config_patch, num_voxels,
        device=device,
        pretrain_root=ckpt_config.pretrain_gm_path,
        logger=None,
        ddim_steps=config.ddim_steps,
        global_pool=ckpt_config.global_pool,
        use_time_cond=ckpt_config.use_time_cond,
    )
    generative_model.model.load_state_dict(sd["model_state_dict"], strict=False)
    print("Loaded eLDM base model")

    # --- optionally inject fine-tuned encoder ---
    if config.inject_encoder and config.encoder_path is not None:
        enc_sd = load_full(config.encoder_path, map_location="cpu")
        enc_key = "model" if "model" in enc_sd else "model_state_dict"
        enc_weights = enc_sd[enc_key]
        prefixed = {"cond_stage_model.mae." + k: v for k, v in enc_weights.items()}
        m, u = generative_model.model.load_state_dict(prefixed, strict=False)
        print(f"Injected fine-tuned encoder ({len(prefixed) - len(u)} keys replaced)")

    # --- generation loop (adapted from eLDM_eval.generate) ---
    model = generative_model.model.to(device)
    ldm_config = generative_model.ldm_config
    shape = (
        ldm_config.model.params.channels,
        ldm_config.model.params.image_size,
        ldm_config.model.params.image_size,
    )
    sampler = PLMSSampler(model)
    num_samples = config.num_samples
    batch_accel = config.batch_size_accel

    os.makedirs(output_dir, exist_ok=True)

    with model.ema_scope():
        model.eval()
        for idx in range(len(dataset_test)):
            item = dataset_test[idx]
            image_name = get_image_name(dataset_test, idx)
            sample_dir = os.path.join(output_dir, image_name)
            os.makedirs(sample_dir, exist_ok=True)

            gt_path = get_gt_image_path(config.imagenet_path, image_name)
            gt_img = Image.open(gt_path).convert("RGB").resize((512, 512), Image.LANCZOS)
            gt_img.save(os.path.join(sample_dir, "GT.png"))

            cap_file = caption_path(config.text_data, image_name)
            if os.path.isfile(cap_file):
                with open(cap_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                text = ""
            with open(os.path.join(sample_dir, "text.txt"), "w", encoding="utf-8") as f:
                f.write(text)

            latent = item["eeg"]
            c, _ = model.get_learned_conditioning(
                repeat(latent, "h w -> c h w", c=batch_accel).to(device)
            )
            samples_ddim, _ = sampler.sample(
                S=config.ddim_steps,
                conditioning=c,
                batch_size=batch_accel,
                shape=shape,
                verbose=False,
            )
            samples_ddim = samples_ddim[:num_samples]
            x_samples = model.decode_first_stage(samples_ddim)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

            for j in range(num_samples):
                img_np = (255.0 * rearrange(x_samples[j], "c h w -> h w c").cpu().numpy()).astype(np.uint8)
                Image.fromarray(img_np).save(os.path.join(sample_dir, f"{j + 1}.png"))

            print(f"[{idx + 1}/{len(dataset_test)}] {image_name}  ({num_samples} images saved)")

    model.to("cpu")
    print(f"\nDone. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
