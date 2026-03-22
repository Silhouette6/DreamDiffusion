"""Trusted torch.load for pickled checkpoints/data (PyTorch 2.6+ defaults weights_only=True)."""
import torch


def load_full(path, map_location=None):
    kw = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kw)
    except TypeError:
        return torch.load(path, **kw)
