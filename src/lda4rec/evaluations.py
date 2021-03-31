# -*- coding: utf-8 -*-
"""
Functions to evaluate a trained model

Note: Some code was taken from Spotlight (MIT)
"""

import numpy as np
import pandas as pd
import scipy.stats as st

FLOAT_MAX = np.finfo(np.float32).max


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


def auc_score(model, test, train=None, rng=None):
    """Calculate AUC Score"""
    test = test.to_csr()

    if train is not None:
        train = train.to_csr()

    rng = np.random.default_rng(rng)

    auc_score = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        # Make predictions for all items
        predictions = model.predict(user_id)

        pos_targets = row.indices

        if train is not None:
            skip_items = train[user_id].indices
            # remove all elements from pos_target which are already in skip_items
            pos_targets = pos_targets[np.in1d(pos_targets, skip_items, invert=True)]

        n_preds = len(pos_targets)
        neg_targets = np.setdiff1d(np.arange(len(predictions)), pos_targets)
        neg_targets = rng.choice(neg_targets, size=n_preds, replace=False)

        # Obtain predictions for all positives
        pos_predictions = predictions[pos_targets]

        # Obtain predictions for random set of unobserved that has the same length
        # as the positives
        neg_predictions = predictions[neg_targets]

        # Compare both ratings for ranking distortions, i.e. positive < negative
        user_auc_score = (pos_predictions > neg_predictions).sum() / n_preds

        auc_score.append(user_auc_score)

    return np.array(auc_score)


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.

    Args:
        model: fitted instance of a recommender model
            The model to evaluate.
        test: :class:`spotlight.interactions.Interactions`
            Test interactions.

    Returns:
        float: nThe RMSE score.
    """
    predictions = model.predict(test.user_ids, test.item_ids)
    ratings = np.clip(test.ratings_batch, 0, 1)  # bring -1 to 0

    return np.sqrt(((ratings - predictions) ** 2).mean())


def summary(
    est,
    *,
    train: np.ndarray,
    test: np.ndarray,
    valid: np.ndarray = None,
    rng=None,
) -> pd.DataFrame:
    """Summarize all metrics in one DataFrame for train/valid/test"""

    def eval(name, test, train=None, rng=None):
        prec, recall = [
            vals.mean() for vals in precision_recall_score(est, test, train)
        ]
        mrr = mrr_score(est, test, train).mean()
        auc = auc_score(est, test, train, rng).mean()
        return pd.Series(
            dict(prec=prec, recall=recall, mrr=mrr, auc=auc), name=name
        ).to_frame()

    rng = np.random.default_rng(rng)

    res_list = []
    train_res = eval("train", train, rng=rng)
    res_list.append(train_res)
    if valid is not None:
        valid_res = eval("valid", valid, train, rng=rng)
        res_list.append(valid_res)
    test_res = eval("test", test, train, rng=rng)
    res_list.append(test_res)

    res = pd.concat(res_list, axis=1)
    res = res.rename_axis("metric")
    return res
