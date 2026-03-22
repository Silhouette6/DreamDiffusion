"""
Print EEG train/test split lengths (same logic as dataset.Splitter + EEGDataset subject filter).
Does not import dataset.py (avoids CLIP/HF download and scipy).

From repo root, after:
  conda activate eeg
run:
  python code/check_eeg_split_lens.py
Optional:
  python code/check_eeg_split_lens.py --eeg datasets/eeg_5_95_std.pth --splits datasets/block_splits_by_image_single.pth
"""
import argparse
import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

ROOT = os.path.dirname(CODE_DIR)

from torch_compat import load_full


def split_lengths(eeg_signals_path, splits_path, subject=4, split_num=0):
    loaded_eeg = load_full(eeg_signals_path)
    if subject != 0:
        data = [
            loaded_eeg["dataset"][i]
            for i in range(len(loaded_eeg["dataset"]))
            if loaded_eeg["dataset"][i]["subject"] == subject
        ]
    else:
        data = loaded_eeg["dataset"]

    loaded_split = load_full(splits_path)
    splits = loaded_split["splits"][split_num]

    def filt(name):
        raw = splits[name]
        return [
            i
            for i in raw
            if i <= len(data)
            and 450 <= data[i]["eeg"].size(1) <= 600
        ]

    return len(filt("train")), len(filt("test"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eeg",
        default=os.path.join(ROOT, "datasets", "eeg_5_95_std.pth"),
    )
    p.add_argument(
        "--splits",
        default=os.path.join(ROOT, "datasets", "block_splits_by_image_single.pth"),
    )
    p.add_argument("--subject", type=int, default=4)
    args = p.parse_args()

    n_tr, n_te = split_lengths(args.eeg, args.splits, subject=args.subject)
    cap = min(10, n_tr)
    print(f"len(train split): {n_tr}")
    print(f"len(test split):  {n_te}")
    print(f"gen_eval limit=10 -> train trials used: {cap}")
    print(f"gen_eval total PLMS runs (train + test): {cap + n_te}")
    num_samples = 25
    print(f"generated images @ num_samples={num_samples}: {num_samples * (cap + n_te)}")


if __name__ == "__main__":
    main()
