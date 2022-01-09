# -*- coding: utf-8 -*-
"""
Functions to evaluate a trained model

Copyright (C) 2021 Florian Wilhelm
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU Affero General Public License as published by the
Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>
"""
from functools import partial

import numpy as np
import pandas as pd
import torch
from rankereval import BinaryLabels, Precision, Rankings, Recall, ReciprocalRank
from scipy.stats import entropy, kendalltau

from . import estimators
from .datasets import Interactions, get_dataset, items_per_user_train_test_split
from .utils import Config, split_along_dim_apply

FLOAT_MAX = np.finfo(np.float32).max


def calc_preds(est, test, train=None) -> np.array:
    test = test.to_csr()
    if train is not None:
        train = train.to_csr()

    predictions = []
    for user_id, row in enumerate(test):

        if not len(row.indices):
            predictions.append(-FLOAT_MAX * np.ones((1, test.shape[1])))
            continue

        user_preds = est.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            user_preds[rated] = -FLOAT_MAX

        predictions.append(user_preds[np.newaxis])
    return np.concatenate(predictions)


def summary(
    est,
    *,
    train: Interactions,
    test: Interactions,
    valid: Interactions = None,
    k: int = 10,
) -> pd.DataFrame:
    """Summarize all metrics in one DataFrame for train/valid/test"""

    def score(name, labels, ranks):
        recall = Recall(k).mean(labels, ranks)["score"]
        prec = Precision(k).mean(labels, ranks)["score"]
        mrr = ReciprocalRank(k).mean(labels, ranks)["score"]
        return pd.Series(dict(prec=prec, recall=recall, mrr=mrr), name=name).to_frame()

    train_ranks = Rankings.from_scores(calc_preds(est, train))
    test_ranks = Rankings.from_scores(calc_preds(est, test, train))
    train_labels = BinaryLabels.from_sparse(train.to_coo())
    test_labels = BinaryLabels.from_sparse(test.to_coo())

    res_list = [score("train", train_labels, train_ranks)]

    if valid is not None:
        valid_ranks = Rankings.from_scores(calc_preds(est, valid, train))
        valid_labels = BinaryLabels.from_sparse(valid.to_coo())
        res_list.append(score("valid", valid_labels, valid_ranks))

    res_list.append(score("test", test_labels, test_ranks))

    res = pd.concat(res_list, axis=1)
    res = res.rename_axis("metric")
    return res


def norm_entropy(a: torch.Tensor) -> float:
    """Normed entropy"""
    n = torch.ones_like(a)
    n = n / n.sum()
    return entropy(a) / entropy(n)


def dist_overlap(a: torch.Tensor, b: torch.Tensor, eps=1e-4) -> float:
    """Overlap of two categorical distributions"""
    assert ((a.sum() - 1).abs() < eps) and ((b.sum() - 1).abs() < eps)
    a, b = a.expand(1, -1), b.expand(1, -1)
    return torch.cat([a, b], dim=0).min(dim=0).values.sum().item()


def cohort_user_interaction_log_probs(
    data, user_dists: torch.Tensor, cohort_dists: torch.Tensor, n_users=None, rng=None
):
    """Calculate log probabilities between a cohort and the user's interactions

    For each user take the cohort with maximal preference of the user and the lowest.
    Using the user's interaction calculate the log prob of seeing those interaction
    using the probs from the cohort with lowest and highest preference of the user.
    """

    def _get_low_high_pref(user_dist: torch.Tensor):
        high = user_dist.argmax().item()
        low = rng.choice((user_dist == 0.0).nonzero().squeeze())
        return low, high

    df = data.to_pandas().set_index("user_id")
    rng = np.random.default_rng(rng)
    user_ids = np.arange(user_dists.shape[0])
    if n_users is not None:
        user_ids = rng.choice(user_ids, n_users, replace=False)

    log_probs = []
    for user_id in user_ids:
        item_ids = torch.from_numpy(df["item_id"].loc[user_id].values).type(torch.long)
        low, high = _get_low_high_pref(user_dists[user_id])
        log_prob_low = torch.log(cohort_dists[item_ids, low]).sum()
        log_prob_high = torch.log(cohort_dists[item_ids, high]).sum()
        log_probs.append([log_prob_low, log_prob_high])

    log_probs = torch.tensor(log_probs)
    log_probs[torch.isinf(log_probs)] = -1e16  # just small enough but not too tiny ;-)
    return user_ids, log_probs


def get_empirical_pops(data):
    """Calculates empirical popularity by just counting the interactions

    Makes sure that items, which are not interacted with, get 0 as popularity
    """
    df = data.to_pandas()
    emp_pops = np.zeros(data.n_items, dtype=int)
    val_counts = df["item_id"].value_counts()
    emp_pops[val_counts.index] = val_counts.to_numpy()
    return torch.from_numpy(emp_pops)


def popularity_ranking_corr(data, pop_dist):
    """Compare popularity ranking to empirical popularity"""
    emp_pop = get_empirical_pops(data)
    return kendalltau(pop_dist, emp_pop)


def conformity_interaction_pop_ranking_corr(pop, confs, data):
    """Compare the conformity of users to the mean popularity
    of the items interacted with.

    Use empirical pop or determined popularity as `pop`
    """
    pop = pop.numpy()
    df = data.to_pandas()
    user_interaction_pops = df.groupby("user_id").apply(
        lambda grp: pop[grp["item_id"].values].mean()
    )
    return kendalltau(
        user_interaction_pops, confs[user_interaction_pops.index].flatten()
    )


def find_good_bad_rnd_twins(user_dists: torch.Tensor, n_users=None, rng=None):
    """Finds for each user a twin user that has the most similar cohort distribution,
    a bad twin that is most dissimilar and a random twin.

    Consider only n users if `n_users` is given.

    Returns user_idx for indexing later and the twins
    """
    rng = np.random.default_rng(rng)

    user_ids = np.arange(user_dists.shape[0])
    if n_users is not None:
        user_ids = rng.choice(user_ids, n_users, replace=False)

    all_user_ids = set(np.arange(user_dists.shape[0]))
    good_twins = []
    bad_twins = []
    rnd_twins = []
    for user_id in user_ids:
        neighbors = split_along_dim_apply(
            user_dists, partial(dist_overlap, user_dists[user_id]), dim=0
        )
        neighbors[user_id] = 0.0  # eliminate itself
        good_twins.append(neighbors.argmax().item())
        neighbors[user_id] = 1.0  # eliminate itself
        bad_twins.append(neighbors.argmin().item())
        rnd_twins.append(rng.choice(list(all_user_ids - {user_id})))

    return user_ids, np.array(good_twins), np.array(bad_twins), np.array(rnd_twins)


def get_twin_jacs(user_ids, twins, data):
    """Calculates Jaccard Coefficient of the user's interaction with those of
    the twins'."""
    assert twins.shape == user_ids.shape

    df = data.to_pandas().set_index("user_id")
    jac = []
    for c_id, user_id in enumerate(user_ids):
        users_items = set(df["item_id"].loc[user_id].values)
        twins_items = set(df["item_id"].loc[twins[c_id]].values)
        jac.append(len(users_items & twins_items) / len(users_items | twins_items))

    return np.array(jac)


def get_cfgs_from_path(path):
    """Return alls configs in directory"""
    for exp_file in path.glob("exp_*"):
        yield Config(exp_file)


def get_model_path_for_exp(cfg, model_dir):
    """Get corresponding model for config file"""
    exp_name = cfg["main"]["name"]
    model_path = list(model_dir.glob(f"{exp_name}_*"))
    assert model_path, f"No model found in {model_dir}"
    assert len(model_path) == 1, "More than one model found for experiment!"
    return model_path[0]


def load_model(cfg, data):
    """Load a corresponding model given a config file from directory"""
    exp_cfg = cfg["experiment"]
    Model = getattr(estimators, exp_cfg["estimator"])
    model_dir = cfg["main"]["model_path"]
    model_path = get_model_path_for_exp(cfg, model_dir)
    est = Model(**exp_cfg["est_params"])
    est.load(model_path, data)
    return est


def get_train_test_data(cfg):
    """Get train/test data and data_rng using the config."""
    exp_cfg = cfg["experiment"]
    data_rng = np.random.default_rng(exp_cfg["dataset_seed"])
    data = get_dataset(exp_cfg["dataset"], data_dir=cfg["main"]["data_path"])
    data.implicit_(exp_cfg["interaction_pivot"])  # implicit feedback
    data.max_user_interactions_(exp_cfg["max_user_interactions"], rng=data_rng)
    data.min_user_interactions_(exp_cfg["min_user_interactions"])
    assert exp_cfg["train_test_split"] == "items_per_user_train_test_split"

    return (
        *items_per_user_train_test_split(data, n_items_per_user=10, rng=data_rng),
        data_rng,
    )
