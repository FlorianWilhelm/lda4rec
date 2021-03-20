# -*- coding: utf-8 -*-
"""
Various utility functions

Note: Many functions copied over from Spotlight (MIT)
"""
import logging

import numpy as np
import pyro
import torch

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


# ToDo: Check if this needs to be in numpy?
def shuffle(*arrays, **kwargs):
    random_state = kwargs.get("random_state")

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have " "the same length.")

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

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


# ToDo: Check if this needs to be in numpy?
def sample_items(num_items, shape, random_state=None):
    """
    Randomly sample a number of items.

    Parameters
    ----------

    num_items: int
        Total number of items from which we should sample:
        the maximum value of a sampled item id will be smaller
        than this.
    shape: int or tuple of ints
        Shape of the sampled array.
    random_state: np.random.RandomState instance, optional
        Random state to use for sampling.

    Returns
    -------

    items: np.array of shape [shape]
        Sampled item ids.
    """

    if random_state is None:
        random_state = np.random.RandomState()

    items = random_state.randint(0, num_items, shape, dtype=np.int64)

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
