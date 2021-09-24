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
import numpy as np
import pandas as pd
from rankereval import AP, BinaryLabels, Precision, Rankings, Recall, ReciprocalRank

from .datasets import Interactions

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
