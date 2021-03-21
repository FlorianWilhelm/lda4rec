# -*- coding: utf-8 -*-
"""
Various utility functions

Note: Many functions copied over from Spotlight (MIT)
"""
import logging
import os.path
from collections import UserDict
from collections.abc import MutableMapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import neptune
import numpy as np
import pandas as pd
import pyro
import torch
import yaml
from neptune import OfflineBackend

from .datasets import Interactions

_logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def gpu(tensor, gpu=False):
    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):
    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


# ToDo: Check if this needs to be in numpy or better in pytorch?
def shuffle(*arrays, **kwargs):
    rng = np.random.default_rng(kwargs.get("rng"))

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    rng.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed, cuda=False):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


# ToDo: Check if this needs to be in numpy or better pytorch?
def sample_items(num_items, shape, rng=None):
    """Randomly sample a number of items"""
    rng = np.random.default_rng(rng)
    items = rng.integers(0, num_items, shape, dtype=np.int64)
    return items


def process_ids(user_ids, item_ids, n_items, use_cuda, cartesian):
    if item_ids is None:
        item_ids = np.arange(n_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))

    if cartesian:
        item_ids, user_ids = (
            item_ids.repeat(user_ids.size(0), 1),
            user_ids.repeat(1, item_ids.size(0)).view(-1, 1),
        )
    else:
        user_ids = user_ids.expand(item_ids.size(0), 1)

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    return user_var.squeeze(), item_var.squeeze()


def relpath_to_abspath(path: Path, anchor_path: Path):
    if path.is_absolute():
        return path
    return (anchor_path / path).resolve()


class Config(UserDict):
    def __init__(self, path: Path, **kwargs):
        super().__init__()

        with open(path, "r") as fh:
            self.yaml_content = fh.read()

        cfg = yaml.safe_load(self.yaml_content)
        timestamp = datetime.now()
        # store name of config file for id later
        cfg["main"].setdefault("name", os.path.splitext(path.name)[0])
        cfg["main"]["log_level"] = getattr(logging, cfg["main"]["log_level"])
        cfg["main"]["timestamp"] = timestamp
        cfg["main"]["timestamp_str"] = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        cfg["main"].update(kwargs)
        self._resolve_paths(cfg, path.parent)

        sec_cfg = cfg["neptune"]["init"]
        if sec_cfg["api_token"].upper() != "ANONYMOUS":
            # read token file and replace it in config
            with open(Path(sec_cfg["api_token"]).expanduser()) as fh:
                sec_cfg["api_token"] = fh.readline().strip()
        if not cfg["neptune"]["activate"]:
            sec_cfg["backend"] = OfflineBackend()

        self.data.update(cfg)  # set cfg as own dictionary

    def _resolve_paths(self, cfg: Dict[str, Any], anchor_path: Path):
        """Resolve all relative paths using `anchor_path` inplace"""
        for k, v in cfg.items():
            if isinstance(v, dict):
                self._resolve_paths(v, anchor_path)
            elif k.endswith("_path"):
                cfg[k] = relpath_to_abspath(Path(v).expanduser(), anchor_path)


class DuplicateKeyError(ValueError):
    pass


def flatten_dict(d, flat_dict=None):
    """Flattens the dict by keeping only the most nested keys & raises if ambiguous"""
    if flat_dict is None:
        flat_dict = {}

    for k, v in d.items():
        if isinstance(v, MutableMapping):
            flatten_dict(v, flat_dict)
        elif k in flat_dict:
            raise DuplicateKeyError(f"Duplicate key: {k}")
        else:
            flat_dict[k] = v
    return flat_dict


def log_summary(df: pd.DataFrame):
    for _, row in df.iterrows():
        metric = row["metric"]
        neptune.log_metric(f"{metric}_train", row["train"])
        neptune.log_metric(f"{metric}_valid", row["valid"])
        neptune.log_metric(f"{metric}_test", row["test"])


def log_dataset(name, interactions: Interactions):
    neptune.set_property(f"{name}_hash", interactions.hash())
    # we count the actual unique entities in the dataset!
    for prop_name, prop_val in [
        ("n_users", len(np.unique(interactions.user_ids))),
        ("n_items", len(np.unique(interactions.item_ids))),
        ("n_interactions", len(interactions)),
    ]:
        neptune.set_property(f"{name}_{prop_name}", prop_val)
