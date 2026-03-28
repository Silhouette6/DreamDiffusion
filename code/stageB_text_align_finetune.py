"""
Stage B - Text-Aligned EEG Encoder Fine-tuning
Triple-modality (EEG-Image-Text) joint supervision with:
  L_vis  : MSE visual alignment (unit-sphere)
  L_txt  : Symmetric InfoNCE textual alignment
  L_cons : MSE consistency regularization (unit-sphere)
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config_TextAlign_Finetune
from sc_mbm.mae_for_eeg import eeg_encoder
from torch_compat import load_full


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGEmbeddingDataset(Dataset):
    """EEG dataset that pre-loads CLIP image/text embeddings into memory."""

    def __init__(self, eeg_signals_path, splits_path, image_embed_dir, text_embed_dir,
                 split_num=0, split_name="train", subject=4):
        loaded = load_full(eeg_signals_path, map_location="cpu")

        if subject != 0:
            self.data = [s for s in loaded["dataset"] if s["subject"] == subject]
        else:
            self.data = loaded["dataset"]

        self.images = loaded["images"]
        self.data_len = 512

        splits = load_full(splits_path, map_location="cpu")
        split_indices = splits["splits"][split_num][split_name]

        self.samples = []
        skipped = 0
        for idx in split_indices:
            if idx < 0 or idx >= len(self.data):
                continue
            sample = self.data[idx]
            eeg_raw = sample["eeg"]
            try:
                tlen = int(eeg_raw.size(1))
            except Exception:
                tlen = int(eeg_raw.shape[1])
            if not (450 <= tlen <= 600):
                continue

            image_name = self.images[self._as_int(sample["image"])]
            wnid = image_name.split("_")[0]

            img_emb_path = os.path.join(image_embed_dir, wnid, f"{image_name}.pt")
            txt_emb_path = os.path.join(text_embed_dir, wnid, f"{image_name}.pt")
            if not os.path.isfile(img_emb_path) or not os.path.isfile(txt_emb_path):
                skipped += 1
                continue

            img_emb = load_full(img_emb_path, map_location="cpu")["embedding"].float()
            txt_emb = load_full(txt_emb_path, map_location="cpu")["embedding"].float()

            img_emb = F.normalize(img_emb, dim=-1)
            txt_emb = F.normalize(txt_emb, dim=-1)

            label = int(sample["label"]) if not isinstance(sample["label"], int) else sample["label"]

            self.samples.append({
                "eeg_raw": eeg_raw,
                "image_embed": img_emb,
                "text_embed": txt_emb,
                "label": label,
            })

        print(f"[{split_name}] loaded {len(self.samples)} samples, skipped {skipped} (missing embeddings)")

    @staticmethod
    def _as_int(x):
        if isinstance(x, int):
            return x
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        eeg = s["eeg_raw"].float().t()          # (T, C)
        eeg = eeg[20:460, :]                     # crop temporal window
        eeg = eeg.transpose(0, 1).numpy()          # (C, T')
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()       # (128, 512)

        return {
            "eeg": eeg,
            "image_embed": s["image_embed"],
            "text_embed": s["text_embed"],
            "label": torch.tensor(s["label"]).long(),
        }


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """Conv1d(seq_len->1) + LayerNorm + GELU + Dropout + Linear(embed->clip_dim)"""

    def __init__(self, seq_len=128, embed_dim=1024, clip_dim=768, dropout=0.5):
        super().__init__()
        self.pool = nn.Conv1d(seq_len, 1, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, clip_dim, bias=True)

    def forward(self, x):
        # x: (N, seq_len, embed_dim)
        x = self.pool(x).squeeze(1)   # (N, embed_dim)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc(x)                # (N, clip_dim)
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextAlignModel(nn.Module):

    def __init__(self, encoder, seq_len, embed_dim, clip_dim,
                 dropout=0.5, temperature_init=0.07, learnable_temperature=True):
        super().__init__()
        self.encoder = encoder
        self.proj_vis = ProjectionHead(seq_len, embed_dim, clip_dim, dropout)
        self.proj_txt = ProjectionHead(seq_len, embed_dim, clip_dim, dropout)
        log_temp = math.log(1.0 / temperature_init)
        self.log_temperature = nn.Parameter(
            torch.tensor(log_temp),
            requires_grad=learnable_temperature,
        )

    @property
    def temperature(self):
        return torch.exp(-self.log_temperature).clamp(min=1e-4)

    def forward(self, eeg):
        latent = self.encoder(eeg)                   # (N, seq_len, embed_dim)
        z_vis = self.proj_vis(latent)                 # (N, clip_dim)
        z_txt = self.proj_txt(latent)                 # (N, clip_dim)
        return z_vis, z_txt


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def loss_vis(z_vis, img_embeds):
    """MSE on L2-normalised vectors (equivalent to 2 - 2*cos_sim)."""
    z_vis_n = F.normalize(z_vis, dim=-1)
    return F.mse_loss(z_vis_n, img_embeds)


def loss_txt_symmetric(z_txt, txt_embeds, temperature):
    """Symmetric InfoNCE (CLIP-style bidirectional contrastive)."""
    z_txt_n = F.normalize(z_txt, dim=-1)
    # (N, N) cosine similarity matrix
    logits = z_txt_n @ txt_embeds.t() / temperature
    N = logits.size(0)
    labels = torch.arange(N, device=logits.device)
    loss_e2t = F.cross_entropy(logits, labels)
    loss_t2e = F.cross_entropy(logits.t(), labels)
    return (loss_e2t + loss_t2e) / 2.0


def loss_cons(z_vis, z_txt):
    """Consistency regularization: MSE between L2-normalised projection outputs."""
    z_vis_n = F.normalize(z_vis, dim=-1)
    z_txt_n = F.normalize(z_txt, dim=-1)
    return F.mse_loss(z_vis_n, z_txt_n)


# ---------------------------------------------------------------------------
# Freeze / parameter groups
# ---------------------------------------------------------------------------

def setup_freeze_and_param_groups(model, config):
    """Freeze encoder layers and return differential-LR parameter groups."""
    encoder = model.encoder
    depth = config.depth
    num_unfreeze = config.num_unfreeze_blocks
    freeze_up_to = depth - num_unfreeze

    # freeze patch_embed
    for p in encoder.patch_embed.parameters():
        p.requires_grad = False
    # freeze pos_embed (already requires_grad=False by default, be explicit)
    encoder.pos_embed.requires_grad = False
    # freeze cls_token & mask_token
    encoder.cls_token.requires_grad = False
    encoder.mask_token.requires_grad = False

    # freeze blocks [0, freeze_up_to)
    for i in range(freeze_up_to):
        for p in encoder.blocks[i].parameters():
            p.requires_grad = False

    # unfrozen encoder params: blocks[freeze_up_to:] + norm
    encoder_params = []
    for i in range(freeze_up_to, depth):
        encoder_params.extend(encoder.blocks[i].parameters())
    encoder_params.extend(encoder.norm.parameters())

    # head params: proj_vis + proj_txt + log_temperature
    head_params = list(model.proj_vis.parameters()) + list(model.proj_txt.parameters())
    if model.log_temperature.requires_grad:
        head_params.append(model.log_temperature)

    param_groups = [
        {"params": encoder_params, "lr": config.lr_encoder},
        {"params": head_params,    "lr": config.lr_heads},
    ]
    return param_groups


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def cosine_warmup_lr(epoch, warmup_epochs, total_epochs, base_lr, min_lr=0.0):
    if epoch < warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def adjust_lr(optimizer, epoch, config):
    lr_enc = cosine_warmup_lr(epoch, config.warmup_epochs, config.num_epoch,
                              config.lr_encoder)
    lr_head = cosine_warmup_lr(epoch, config.warmup_epochs, config.num_epoch,
                               config.lr_heads)
    optimizer.param_groups[0]["lr"] = lr_enc
    optimizer.param_groups[1]["lr"] = lr_head
    return lr_enc, lr_head


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, config, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_ep{epoch:03d}.pth")
    torch.save({
        "model": model.encoder.state_dict(),
        "config": config,
        "epoch": epoch,
        "proj_vis": model.proj_vis.state_dict(),
        "proj_txt": model.proj_txt.state_dict(),
    }, path)
    latest = os.path.join(output_dir, "checkpoint.pth")
    torch.save({
        "model": model.encoder.state_dict(),
        "config": config,
        "epoch": epoch,
        "proj_vis": model.proj_vis.state_dict(),
        "proj_txt": model.proj_txt.state_dict(),
    }, latest)
    print(f"  [save] {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, config, device):
    model.eval()
    total_vis = total_txt = total_cons = total_all = 0.0
    n = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        img_emb = batch["image_embed"].to(device)
        txt_emb = batch["text_embed"].to(device)

        z_vis, z_txt = model(eeg)

        lv = loss_vis(z_vis, img_emb)
        lt = loss_txt_symmetric(z_txt, txt_emb, model.temperature)
        lc = loss_cons(z_vis, z_txt)
        lt_total = config.lambda_vis * lv + config.lambda_txt * lt + config.lambda_cons * lc

        bs = eeg.size(0)
        total_vis += lv.item() * bs
        total_txt += lt.item() * bs
        total_cons += lc.item() * bs
        total_all += lt_total.item() * bs
        n += bs

    model.train()
    return {
        "loss": total_all / max(n, 1),
        "vis": total_vis / max(n, 1),
        "txt": total_txt / max(n, 1),
        "cons": total_cons / max(n, 1),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data ---
    train_ds = EEGEmbeddingDataset(
        config.eeg_signals_path, config.splits_path,
        config.image_embed_dir, config.text_embed_dir,
        split_name="train", subject=config.subject,
    )
    test_ds = EEGEmbeddingDataset(
        config.eeg_signals_path, config.splits_path,
        config.image_embed_dir, config.text_embed_dir,
        split_name="test", subject=config.subject,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    # --- encoder from MAE pretrain ---
    pretrain = load_full(config.pretrain_mbm_path, map_location="cpu")
    num_voxels = train_ds.data_len  # 512
    seq_len = num_voxels // config.patch_size  # 128

    encoder = eeg_encoder(
        time_len=num_voxels,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        global_pool=config.global_pool,
    )
    sd_key = "model" if "model" in pretrain else "model_state_dict"
    raw_sd = pretrain[sd_key]

    # If checkpoint is a full eLDM model, extract EEG encoder weights
    # by stripping the 'cond_stage_model.mae.' prefix.
    prefix = "cond_stage_model.mae."
    if any(k.startswith(prefix) for k in raw_sd.keys()):
        raw_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
        print(f"Extracted EEG encoder weights from full eLDM checkpoint (prefix='{prefix}', {len(raw_sd)} keys)")

    encoder.load_checkpoint(raw_sd)
    print("Loaded EEG encoder from pretrain checkpoint")

    # --- model ---
    model = TextAlignModel(
        encoder, seq_len, config.embed_dim, config.clip_dim,
        dropout=config.proj_dropout,
        temperature_init=config.temperature_init,
        learnable_temperature=config.learnable_temperature,
    ).to(device)

    # --- freeze + optimizer ---
    param_groups = setup_freeze_and_param_groups(model, config)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(config.use_amp and device.type == "cuda"))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # --- training loop ---
    run_dir = os.path.join(config.output_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = os.path.join(run_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"Experiment dir: {run_dir}")
    print(f"TensorBoard logs: {tb_dir}")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, config.num_epoch + 1):
        model.train()
        lr_enc, lr_head = adjust_lr(optimizer, epoch - 1, config)

        epoch_vis = epoch_txt = epoch_cons = epoch_total = 0.0
        n_samples = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            eeg = batch["eeg"].to(device, non_blocking=True)
            img_emb = batch["image_embed"].to(device, non_blocking=True)
            txt_emb = batch["text_embed"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(config.use_amp and device.type == "cuda")):
                z_vis, z_txt = model(eeg)
                lv = loss_vis(z_vis, img_emb)
                lt = loss_txt_symmetric(z_txt, txt_emb, model.temperature)
                lc = loss_cons(z_vis, z_txt)
                total_loss = config.lambda_vis * lv + config.lambda_txt * lt + config.lambda_cons * lc

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            if config.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.clip_grad,
                )
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            bs = eeg.size(0)
            epoch_vis += lv.item() * bs
            epoch_txt += lt.item() * bs
            epoch_cons += lc.item() * bs
            epoch_total += total_loss.item() * bs
            n_samples += bs

            # per-step TensorBoard logging
            if step % config.log_every_n_step == 0:
                writer.add_scalar("train_step/L_vis", lv.item(), global_step)
                writer.add_scalar("train_step/L_txt", lt.item(), global_step)
                writer.add_scalar("train_step/L_cons", lc.item(), global_step)
                writer.add_scalar("train_step/L_total", total_loss.item(), global_step)
                writer.add_scalar("train_step/temperature", model.temperature.item(), global_step)
                print(f"  ep {epoch} step {step}/{len(train_loader)}  "
                      f"L_vis={lv.item():.4f}  L_txt={lt.item():.4f}  "
                      f"L_cons={lc.item():.4f}  tau={model.temperature.item():.4f}")

        dt = time.time() - t0
        avg = lambda x: x / max(n_samples, 1)

        # per-epoch TensorBoard logging (train)
        writer.add_scalar("train_epoch/L_vis", avg(epoch_vis), epoch)
        writer.add_scalar("train_epoch/L_txt", avg(epoch_txt), epoch)
        writer.add_scalar("train_epoch/L_cons", avg(epoch_cons), epoch)
        writer.add_scalar("train_epoch/L_total", avg(epoch_total), epoch)
        writer.add_scalar("lr/encoder", lr_enc, epoch)
        writer.add_scalar("lr/heads", lr_head, epoch)
        writer.add_scalar("params/temperature", model.temperature.item(), epoch)

        print(f"Epoch {epoch}/{config.num_epoch}  "
              f"train: total={avg(epoch_total):.4f} vis={avg(epoch_vis):.4f} "
              f"txt={avg(epoch_txt):.4f} cons={avg(epoch_cons):.4f}  "
              f"lr_enc={lr_enc:.2e} lr_head={lr_head:.2e}  "
              f"tau={model.temperature.item():.4f}  {dt:.1f}s")

        # --- validation ---
        val = validate(model, test_loader, config, device)

        writer.add_scalar("val/L_vis", val["vis"], epoch)
        writer.add_scalar("val/L_txt", val["txt"], epoch)
        writer.add_scalar("val/L_cons", val["cons"], epoch)
        writer.add_scalar("val/L_total", val["loss"], epoch)

        print(f"  val: total={val['loss']:.4f} vis={val['vis']:.4f} "
              f"txt={val['txt']:.4f} cons={val['cons']:.4f}")

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            save_checkpoint(model, config, epoch, run_dir)
            print(f"  ** new best val loss: {best_val_loss:.4f}")

        if epoch % config.save_every_n_epoch == 0:
            save_checkpoint(model, config, epoch, run_dir)

    writer.close()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {run_dir}")
    print(f"TensorBoard logs: {tb_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Text-Aligned EEG Encoder Fine-tuning")

    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--eeg_signals_path", type=str, default=None)
    parser.add_argument("--splits_path", type=str, default=None)
    parser.add_argument("--image_embed_dir", type=str, default=None)
    parser.add_argument("--text_embed_dir", type=str, default=None)
    parser.add_argument("--pretrain_mbm_path", type=str, default=None)

    parser.add_argument("--num_unfreeze_blocks", type=int, default=None)
    parser.add_argument("--proj_dropout", type=float, default=None)

    parser.add_argument("--lambda_vis", type=float, default=None)
    parser.add_argument("--lambda_txt", type=float, default=None)
    parser.add_argument("--lambda_cons", type=float, default=None)
    parser.add_argument("--temperature_init", type=float, default=None)

    parser.add_argument("--lr_heads", type=float, default=None)
    parser.add_argument("--lr_encoder", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--subject", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config_TextAlign_Finetune()

    # override root_path first so dependent paths recalculate
    if args.root_path is not None:
        config.root_path = args.root_path
        config.eeg_signals_path = os.path.join(config.root_path, 'datasets/eeg_5_95_std.pth')
        config.splits_path = os.path.join(config.root_path, 'datasets/block_splits_by_image_single.pth')
        config.imagenet_path = os.path.join(config.root_path, 'datasets/imageNet_images')
        config.image_embed_dir = os.path.join(config.root_path, 'datasets/embeddings/image')
        config.text_embed_dir = os.path.join(config.root_path, 'datasets/embeddings/text')
        config.pretrain_mbm_path = os.path.join(config.root_path, 'pretrains/eeg_pretrain/checkpoint.pth')
        config.output_path = os.path.join(config.root_path, 'exps/text_align/')

    for key in ["output_path", "eeg_signals_path", "splits_path",
                "image_embed_dir", "text_embed_dir", "pretrain_mbm_path",
                "num_unfreeze_blocks", "proj_dropout",
                "lambda_vis", "lambda_txt", "lambda_cons", "temperature_init",
                "lr_heads", "lr_encoder", "weight_decay",
                "num_epoch", "batch_size", "warmup_epochs", "clip_grad",
                "subject", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    if args.no_amp:
        config.use_amp = False

    train(config)


if __name__ == "__main__":
    main()
