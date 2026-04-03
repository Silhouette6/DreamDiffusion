"""
Stage C - Compute per-sample evaluation metrics for generated images.

Input:  a sample folder containing text.txt, GT.png, 1.png, 2.png, 3.png
Output: metrics.json in the same folder with three metrics:
  1. inception_feature_distance  (Inception-v3 L2 feature distance, avg of 3)
  2. top1_accuracy_40way         (40-way zero-shot classification via CLIP)
  3. clip_score_text             (CLIP cosine similarity, avg of 3)
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from config import Config_Metrics


# ---------------------------------------------------------------------------
# Inception Feature Distance
# ---------------------------------------------------------------------------

def _build_inception(device):
    """Load Inception-v3 and return a feature extractor (2048-dim pool output)."""
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inception.fc = torch.nn.Identity()
    inception.eval().to(device)
    return inception


_INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def inception_feature_distance(gt_path, gen_paths, inception, device):
    """Average L2 distance in Inception feature space between GT and generated images."""
    gt_tensor = _INCEPTION_TRANSFORM(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
    gt_feat = inception(gt_tensor).squeeze(0)

    distances = []
    for p in gen_paths:
        img_tensor = _INCEPTION_TRANSFORM(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        feat = inception(img_tensor).squeeze(0)
        dist = torch.norm(gt_feat - feat, p=2).item()
        distances.append(dist)
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def _build_clip(device):
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    return clip_model, processor, tokenizer


@torch.no_grad()
def _encode_images_clip(clip_model, processor, image_paths, device):
    """Return L2-normalised CLIP image embeddings [N, D]."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    feats = clip_model.get_image_features(pixel_values=pixel_values)
    return F.normalize(feats, dim=-1)


@torch.no_grad()
def _encode_texts_clip(clip_model, tokenizer, texts, device):
    """Return L2-normalised CLIP text embeddings [N, D]."""
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    feats = clip_model.get_text_features(**tokens)
    return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# Top-1 Accuracy (40-way zero-shot)
# ---------------------------------------------------------------------------

def _collect_class_captions(text_data_dir):
    """Return dict: wnid -> list of caption strings."""
    text_root = Path(text_data_dir)
    class_captions = {}
    for wnid_dir in sorted(text_root.iterdir()):
        if not wnid_dir.is_dir():
            continue
        wnid = wnid_dir.name
        caps = []
        for cap_file in wnid_dir.glob("*_caption.txt"):
            text = cap_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                caps.append(text)
        if caps:
            class_captions[wnid] = caps
    return class_captions


@torch.no_grad()
def top1_accuracy_40way(gen_paths, target_text, target_wnid,
                        class_captions, clip_model, processor, tokenizer, device):
    """
    40-way zero-shot classification accuracy averaged over generated images.
    For each of the 39 non-target classes, randomly select one caption.
    """
    other_wnids = [w for w in class_captions if w != target_wnid]
    texts = [target_text]
    for w in other_wnids:
        texts.append(random.choice(class_captions[w]))

    text_feats = _encode_texts_clip(clip_model, tokenizer, texts, device)
    img_feats = _encode_images_clip(clip_model, processor, gen_paths, device)

    sims = img_feats @ text_feats.t()
    correct = (sims.argmax(dim=1) == 0).float()
    return float(correct.mean().item())


# ---------------------------------------------------------------------------
# CLIP Score Text
# ---------------------------------------------------------------------------

@torch.no_grad()
def clip_score_text(gen_paths, text, clip_model, processor, tokenizer, device):
    """Average cosine similarity between generated images and text."""
    img_feats = _encode_images_clip(clip_model, processor, gen_paths, device)
    text_feats = _encode_texts_clip(clip_model, tokenizer, [text], device)
    sims = (img_feats @ text_feats.t()).squeeze(1)
    return float(sims.mean().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Stage C: Per-sample image metrics (CLI overrides Config_Metrics)")
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--text_data", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _find_gen_images(folder):
    """Find all numbered generated images (1.png, 2.png, ...) in a folder."""
    i = 1
    paths = []
    while (folder / f"{i}.png").exists():
        paths.append(folder / f"{i}.png")
        i += 1
    return paths


def _is_sample_folder(folder):
    """Check if a folder contains the expected per-sample files."""
    return (folder / "GT.png").exists() and (folder / "text.txt").exists() and len(_find_gen_images(folder)) > 0


def process_one_sample(folder, device, inception, clip_model, processor, tokenizer, class_captions):
    """Compute metrics for a single sample folder. Returns metrics dict."""
    gt_path = folder / "GT.png"
    gen_paths = _find_gen_images(folder)
    text_file = folder / "text.txt"

    text = text_file.read_text(encoding="utf-8", errors="ignore").strip()
    image_name = folder.name
    target_wnid = image_name.split("_")[0]

    ifd = inception_feature_distance(str(gt_path), [str(p) for p in gen_paths], inception, device)

    cs = clip_score_text([str(p) for p in gen_paths], text,
                         clip_model, processor, tokenizer, device)

    acc = top1_accuracy_40way(
        [str(p) for p in gen_paths], text, target_wnid,
        class_captions, clip_model, processor, tokenizer, device,
    )

    metrics = {
        "inception_feature_distance": round(ifd, 6),
        "top1_accuracy_40way": round(acc, 6),
        "clip_score_text": round(cs, 6),
    }
    out_path = folder / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main():
    config = Config_Metrics()
    args = parse_args()

    for key in ["folder", "text_data", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    assert config.folder is not None, "folder must be set in Config_Metrics or via --folder"

    random.seed(config.seed)
    np.random.seed(config.seed)

    root_folder = Path(config.folder)
    assert root_folder.is_dir(), f"Folder not found: {root_folder}"

    # Determine mode: single sample folder vs. parent folder with many samples
    if _is_sample_folder(root_folder):
        sample_folders = [root_folder]
    else:
        sample_folders = sorted(
            d for d in root_folder.iterdir() if d.is_dir() and _is_sample_folder(d)
        )
        assert len(sample_folders) > 0, f"No valid sample folders found in {root_folder}"
        print(f"Batch mode: found {len(sample_folders)} sample folders in {root_folder}")

    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Load models once, reuse for all samples
    print("Loading Inception-v3 ...")
    inception = _build_inception(device)
    print("Loading CLIP model ...")
    clip_model, processor, tokenizer = _build_clip(device)
    print("Collecting class captions ...")
    class_captions = _collect_class_captions(config.text_data)
    print(f"  Found {len(class_captions)} classes in {config.text_data}\n")

    all_metrics = {}
    for i, folder in enumerate(sample_folders):
        image_name = folder.name
        print(f"[{i + 1}/{len(sample_folders)}] {image_name} ...", end="  ")
        m = process_one_sample(folder, device, inception, clip_model, processor, tokenizer, class_captions)
        all_metrics[image_name] = m
        print(f"IFD={m['inception_feature_distance']:.4f}  "
              f"Top1={m['top1_accuracy_40way']:.4f}  "
              f"CLIP={m['clip_score_text']:.4f}")

    del inception, clip_model, processor, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # If batch mode, write aggregate summary to the parent folder
    if len(sample_folders) > 1:
        avg_ifd = np.mean([m["inception_feature_distance"] for m in all_metrics.values()])
        avg_acc = np.mean([m["top1_accuracy_40way"] for m in all_metrics.values()])
        avg_cs = np.mean([m["clip_score_text"] for m in all_metrics.values()])
        summary = {
            "num_samples": len(sample_folders),
            "avg_inception_feature_distance": round(float(avg_ifd), 6),
            "avg_top1_accuracy_40way": round(float(avg_acc), 6),
            "avg_clip_score_text": round(float(avg_cs), 6),
            "per_sample": all_metrics,
        }
        summary_path = root_folder / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n{'=' * 60}")
        print(f"Aggregate over {len(sample_folders)} samples:")
        print(f"  avg_inception_feature_distance = {avg_ifd:.4f}")
        print(f"  avg_top1_accuracy_40way        = {avg_acc:.4f}")
        print(f"  avg_clip_score_text            = {avg_cs:.4f}")
        print(f"Summary saved to: {summary_path}")
    else:
        print(f"\nMetrics saved to: {sample_folders[0] / 'metrics.json'}")
        print(json.dumps(all_metrics[sample_folders[0].name], indent=2))


if __name__ == "__main__":
    main()
