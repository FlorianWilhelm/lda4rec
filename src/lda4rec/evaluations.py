# -*- coding: utf-8 -*-
"""
Functions to evaluate a trained model

Note: Some code was taken from Spotlight (MIT)
ToDo: Replace this with https://github.com/microsoft/rankerEval

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
import numpy as np
import pandas as pd
import scipy.stats as st
from rankereval import AP, BinaryLabels, Precision, Rankings, Recall, ReciprocalRank

from .datasets import Interactions

FLOAT_MAX = np.finfo(np.float32).max


def make_label_ranks(y_true, y_pred):
    y_true = BinaryLabels.from_sparse(y_true)
    y_pred = Rankings.from_sparse(y_pred)
    return y_true, y_pred


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Args:
        model: fitted instance of a recommender model
            The model to evaluate.
        test: :class:`spotlight.interactions.Interactions`
            Test interactions.
        train: :class:`spotlight.interactions.Interactions`, optional
            Train interactions. If supplied, scores of known
            interactions will be set to very low values and so not
            affect the MRR.

    Returns:
        numpy array of shape (num_users,): Array of MRR scores for each user in test.
    """

    test = test.to_csr()

    if train is not None:
        train = train.to_csr()

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    n_hit = len(set(predictions).intersection(set(targets)))

    return float(n_hit) / len(predictions), float(n_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Args:
        model: fitted instance of a recommender model
            The model to evaluate.
        test: :class:`spotlight.interactions.Interactions`
            Test interactions.
        train: :class:`spotlight.interactions.Interactions`, optional
            Train interactions. If supplied, scores of known
            interactions will not affect the computed metrics.
        k: int or array of int,
            The maximum number of predicted items

    Returns:
        numpy array of shape (num_users, len(k)): (Precision@k, Recall@k)
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.to_csr()

    if train is not None:
        train = train.to_csr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(
            *[_get_precision_recall(predictions, targets, x) for x in k]
        )

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def summary(
    est,
    *,
    train: np.ndarray,
    test: np.ndarray,
    valid: np.ndarray = None,
) -> pd.DataFrame:
    """Summarize all metrics in one DataFrame for train/valid/test"""

    def eval(name, test, train=None):
        prec, recall = [
            vals.mean() for vals in precision_recall_score(est, test, train)
        ]
        mrr = mrr_score(est, test, train).mean()
        return pd.Series(dict(prec=prec, recall=recall, mrr=mrr), name=name).to_frame()

    res_list = []
    train_res = eval("train", train)
    res_list.append(train_res)
    if valid is not None:
        valid_res = eval("valid", valid, train)
        res_list.append(valid_res)
    test_res = eval("test", test, train)
    res_list.append(test_res)

    res = pd.concat(res_list, axis=1)
    res = res.rename_axis("metric")
    return res


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


def summary_ng(
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
        map = AP(k).mean(labels, ranks)["score"]
        return pd.Series(
            dict(prec=prec, recall=recall, mrr=mrr, map=map), name=name
        ).to_frame()

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
