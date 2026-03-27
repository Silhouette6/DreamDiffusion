import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from tqdm import tqdm

from torch_compat import load_full


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def _image_path_from_name(imagenet_root: Path, image_name: str) -> Path:
    wnid = image_name.split("_")[0]
    return imagenet_root / wnid / f"{image_name}.JPEG"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Generate CLIP image embeddings (per-sample .pt) for DreamDiffusion.")
    p.add_argument("--eeg_pth", type=str, default="datasets/eeg_5_95_std.pth")
    p.add_argument("--splits_pth", type=str, default="datasets/block_splits_by_image_single.pth")
    p.add_argument("--split_name", type=str, default="train", choices=["train", "test"])
    p.add_argument("--split_num", type=int, default=0)
    p.add_argument("--imagenet_root", type=str, default="datasets/imageNet_images")
    p.add_argument("--output_root", type=str, default="datasets/embeddings/image")
    p.add_argument("--manifest_path", type=str, default="datasets/embeddings/image_manifest.json")
    p.add_argument("--missing_path", type=str, default="datasets/embeddings/missing_images.json")
    p.add_argument("--subject", type=int, default=4, help="0 means all subjects; otherwise filter by subject id.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default=_default_device())
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--max_items", type=int, default=0, help="0 means no limit (debug).")
    args = p.parse_args()

    eeg_pth = Path(args.eeg_pth)
    splits_pth = Path(args.splits_pth)
    imagenet_root = Path(args.imagenet_root)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest_path)
    missing_path = Path(args.missing_path)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_full(str(eeg_pth), map_location="cpu")
    splits = load_full(str(splits_pth), map_location="cpu")
    split_indices: List[int] = splits["splits"][args.split_num][args.split_name]

    dataset_list = data["dataset"]
    images_map = data["images"]  # idx -> image_name

    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")

    device = torch.device(args.device)
    if args.dtype == "fp16":
        amp_dtype = torch.float16
    elif args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    model = model.eval().to(device)

    manifest: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    batch_image_names: List[str] = []
    batch_dataset_indices: List[int] = []
    batch_pixel_values: List[torch.Tensor] = []

    def flush_batch():
        nonlocal batch_image_names, batch_dataset_indices, batch_pixel_values, manifest
        if not batch_pixel_values:
            return
        pixel_values = torch.stack(batch_pixel_values, dim=0).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type != "cpu" and amp_dtype != torch.float32)):
            out = model(pixel_values=pixel_values)
            embeds = out.image_embeds  # [B, D]
        embeds = embeds.detach().to("cpu")

        for image_name, ds_idx, emb in zip(batch_image_names, batch_dataset_indices, embeds):
            wnid = image_name.split("_")[0]
            out_dir = output_root / wnid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_name}.pt"
            torch.save(
                {
                    "embedding": emb,
                    "key": image_name,
                    "image_name": image_name,
                    "wnid": wnid,
                    "dataset_index": ds_idx,
                    "split_name": args.split_name,
                    "subject": args.subject,
                    "model": "openai/clip-vit-large-patch14",
                    "modality": "image",
                },
                out_path,
            )
            manifest.append(
                {
                    "key": image_name,
                    "image_name": image_name,
                    "wnid": wnid,
                    "dataset_index": ds_idx,
                    "split_name": args.split_name,
                    "subject": args.subject,
                    "embedding_path": str(out_path.as_posix()),
                }
            )

        batch_image_names = []
        batch_dataset_indices = []
        batch_pixel_values = []

    kept = 0
    for ds_idx in tqdm(split_indices, desc=f"Processing {args.split_name} image embeddings"):
        if args.max_items and kept >= args.max_items:
            break

        if ds_idx < 0 or ds_idx >= len(dataset_list):
            continue
        sample = dataset_list[ds_idx]

        if args.subject != 0 and _as_int(sample.get("subject", -1)) != args.subject:
            continue

        eeg = sample.get("eeg", None)
        if eeg is None:
            continue
        # match Splitter filter in code/dataset.py
        try:
            tlen = int(eeg.size(1))
        except Exception:
            tlen = int(eeg.shape[1])
        if not (450 <= tlen <= 600):
            continue

        image_idx = _as_int(sample["image"])
        image_name = images_map[image_idx]

        img_path = _image_path_from_name(imagenet_root, image_name)
        if not img_path.exists():
            missing.append(
                {
                    "key": image_name,
                    "image_name": image_name,
                    "wnid": image_name.split("_")[0],
                    "dataset_index": ds_idx,
                    "missing_path": str(img_path.as_posix()),
                    "reason": "file_not_found",
                }
            )
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            missing.append(
                {
                    "key": image_name,
                    "image_name": image_name,
                    "wnid": image_name.split("_")[0],
                    "dataset_index": ds_idx,
                    "missing_path": str(img_path.as_posix()),
                    "reason": f"image_error: {type(e).__name__}",
                }
            )
            continue

        inputs = processor(images=img, return_tensors="pt")
        pv = inputs["pixel_values"].squeeze(0)  # [3, H, W]

        batch_image_names.append(image_name)
        batch_dataset_indices.append(ds_idx)
        batch_pixel_values.append(pv)
        kept += 1

        if len(batch_pixel_values) >= args.batch_size:
            flush_batch()

    flush_batch()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump(missing, f, ensure_ascii=False, indent=2)

    print(f"[image] split={args.split_name} subject={args.subject} exported={len(manifest)} missing={len(missing)}")


if __name__ == "__main__":
    main()
