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

from .datasets import Interactions
from .utils import split_along_dim_apply

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


def dist_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Overlap of two categorical distributions"""
    a, b = a.expand(1, -1), b.expand(1, -1)
    return torch.cat([a, b], dim=0).min(dim=0).values.sum().item()


def cohort_top_n_ranking_corr(cohorts_dists: torch.Tensor, n=10, alpha=0.05):
    """From top n items in each cohort calculate the pairwise overlap followed
    by Kendalls'tau on that overlap. Given an significance level alpha
    the Taus are returned an NA otherwise"""
    n_cohorts = cohorts_dists.shape[1]
    top_n = [
        c.argsort(descending=True)[:n].numpy()
        for c in torch.unbind(cohorts_dists, dim=1)
    ]
    overlap = [
        [np.array(list(set(top_n[i]) & set(top_n[j]))) for j in np.arange(n_cohorts)]
        for i in np.arange(n_cohorts)
    ]
    corr_pval = np.array(
        [
            [
                kendalltau(
                    cohorts_dists[:, i][overlap[i][j]],
                    cohorts_dists[:, j][overlap[i][j]],
                )
                for j in np.arange(n_cohorts)
            ]
            for i in np.arange(n_cohorts)
        ]
    )
    corr, pval = corr_pval[..., 0], corr_pval[..., 1]
    corr[pval > alpha] = np.nan
    return corr


def get_empirical_pops(data):
    """Calculates empirical popularity by just counting the interactions

    Makes sure that items, which are not interacted with, get 0 as popularity
    """
    df = data.to_pandas()
    emp_pops = np.zeros(data.n_items, dtype=int)
    val_counts = df["item_id"].value_counts()
    emp_pops[val_counts.index] = val_counts.to_numpy()
    return emp_pops


def popularity_ranking_corr(data, pop_dist):
    """Compare popularity ranking to empirical popularity"""
    emp_pop = get_empirical_pops(data)
    return kendalltau(pop_dist, emp_pop)


def conformity_interaction_pop_ranking_corr(pop, confs, data):
    """Compare the conformity of users to the median popularity
    of the items interacted with.

    Use empirical pop or determined popularity as `pop`
    """
    df = data.to_pandas()
    user_interaction_pops = df.groupby("user_id").apply(
        lambda grp: np.median(pop[grp["item_id"].values])
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
