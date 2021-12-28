# -*- coding: utf-8 -*-
"""
Estimators

Note: Some code was taken from Spotlight (MIT)

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
import logging
from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Optional

import neptune.new as neptune
import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn.functional as F
import torch.optim as optim
from pyro.infer import SVI, JitTraceEnum_ELBO, Predictive, TraceEnum_ELBO
from pyro.optim import ClippedAdam

from . import lda
from .datasets import Interactions
from .losses import adaptive_hinge_loss, bpr_loss, hinge_loss, logistic_loss
from .nets import MFNet, NMFNet, SNMFNet
from .utils import cpu, gpu, minibatch, process_ids, sample_items, set_seed, shuffle

_logger = logging.getLogger(__name__)


class EstimatorMixin(metaclass=ABCMeta):
    def __repr__(self):
        return "<{}: {}>".format(
            self.__class__.__name__,
            repr(self._model),
        )

    @abstractmethod
    def fit(self, interactions):
        pass

    def _predict(self, user_ids, item_ids):
        """Overwrite this if you need to do more then just applying the model"""
        self._model.train(False)
        return self._model(user_ids, item_ids)

    def predict(self, user_ids, item_ids=None, cartesian=False):
        """
        Make predictions: given a user id, compute the recommendation
        scores for items.

        Use this as Mixin to avoid double implementation

        Args:
            user_ids (int or array): If int, will predict the recommendation scores
                for this user for all items in item_ids. If an array, will predict
                scores for all (user, item) pairs defined by user_ids and item_ids.
            item_ids (array, optional): Array containing the item ids for which
                prediction scores are desired. If not supplied, predictions for all
                items will be computed.
            cartesian (bool, optional): Calculate the prediction for each item times
                each user.

        Returns:
            np.array: Predicted scores for all items in item_ids.
        """
        if np.isscalar(user_ids):
            n_users = 1
        else:
            n_users = len(user_ids)
        try:
            user_ids, item_ids = process_ids(
                user_ids, item_ids, self._n_items, self._use_cuda, cartesian
            )
        except RuntimeError as e:
            raise RuntimeError("Maybe you want to set `cartesian=True`?") from e

        out = self._predict(user_ids, item_ids)
        out = cpu(out).detach().numpy()
        if cartesian:
            return out.reshape(n_users, -1)
        else:
            return out.flatten()


class PopEst(EstimatorMixin):
    """Estimator using the popularity"""

    def __init__(self, *, rng=None):
        self.pops = None
        self._n_users = None
        self._n_items = None
        self._use_cuda = False
        # Only for consistence to other estimators
        self._rng = np.random.default_rng(rng)

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    def fit(self, interactions):
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items

        df = interactions.to_pandas()
        pops = (
            df.groupby("item_id", as_index=False)["rating"]
            .agg("count")
            .rename(columns={"rating": "score"})
        )
        all_items = pd.DataFrame({"item_id": np.arange(interactions.n_items)})
        pops = pd.merge(pops, all_items, how="right", on="item_id").fillna(0.0)["score"]
        pops = pops / pops.max()
        self.pops = torch.from_numpy(pops.to_numpy().squeeze())

    def _predict(self, user_ids, item_ids):
        """Overwrite this if you need to do more then just applying the model"""
        assert self.pops is not None
        return self.pops[item_ids]

    def save(self, filename):
        torch.save(self.pops, filename)
        return self

    def load(self, filename, interactions):
        """Load model, interactions are only used to infer metadata"""
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items
        self.pops = torch.load(filename)
        return self


class LDA4RecEst(EstimatorMixin):
    """Estimator using LDA"""

    def __init__(
        self,
        *,
        embedding_dim: int,
        n_iter: int,
        alpha: Optional[float] = None,
        batch_size: Optional[int] = 32,
        learning_rate: float = 0.01,
        use_jit: bool = True,
        use_cuda: bool = False,
        rng=None,
        predict_posterior=False,
        clear_param_store: bool = True,
        log_steps=100,
        model=None,
        pred_model=None,
        guide=None,
        pred_guide=None,
        n_samples=10_000,
    ):
        self._embedding_dim = (
            embedding_dim  # for easier comparison with other estimators
        )
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._use_jit = use_jit
        self._alpha = alpha
        self._use_cuda = use_cuda
        self._clear_param_store = clear_param_store
        self._log_steps = log_steps
        self._rng = np.random.default_rng(rng)
        self._model = lda.model if model is None else model
        self._pred_model = lda.pred_model if pred_model is None else pred_model
        self._guide = lda.guide if guide is None else guide
        self._pred_guide = lda.pred_guide if pred_guide is None else pred_guide

        self._n_samples = n_samples
        set_seed(self._rng.integers(0, 2 ** 32 - 1), cuda=self._use_cuda)

        # Initialized after fit
        self.pops = None
        self.user_topics = None
        self.topic_items = None
        self.params = None  # all inferred params including above
        self._n_users = None
        self._n_items = None
        self._model_params = None  # params used when fitting
        self.predict_posterior = predict_posterior  # extremely slow if True

    def _initialize(self, model_params):
        self._model_params = model_params
        self.params = pyro.get_param_store()
        self.pops = self.params[lda.Param.item_pops_loc].detach()
        self.topic_items = self.params[lda.Param.topic_items_loc].detach()
        self.user_topics = F.softmax(
            self.params[lda.Param.user_topics_logits], dim=-1
        ).detach()
        self.user_pop_devs = self.params[lda.Param.user_pop_devs_loc].detach()

    def fit(self, interactions, clear_params=None):
        run = neptune.get_last_run()
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items
        interactions = torch.tensor(interactions.to_ratings_per_user(), dtype=torch.int)

        clear_params = self._clear_param_store if clear_params is None else clear_params
        if clear_params:
            pyro.clear_param_store()
        pyro.enable_validation(__debug__)

        model_params = dict(
            interactions=interactions,
            n_topics=self._embedding_dim,
            n_users=self._n_users,
            n_items=self._n_items,
            alpha=self._alpha,
            batch_size=self._batch_size,
        )

        Elbo = JitTraceEnum_ELBO if self._use_jit else TraceEnum_ELBO
        elbo = Elbo(max_plate_nesting=2)
        optim = ClippedAdam({"lr": self._learning_rate})
        svi = SVI(self._model, self._guide, optim, elbo)

        for epoch_num in range(self._n_iter):
            epoch_loss = svi.step(**model_params)
            run["train/loss"].log(epoch_loss)
            if epoch_num % self._log_steps == 0:
                _logger.info("Epoch {: >5d}: loss {}".format(epoch_num, epoch_loss))

        epoch_loss = elbo.loss(self._model, self._guide, **model_params)
        self._initialize(model_params)

        return epoch_loss

    def _predict_posterior(self, user_ids, item_ids):
        """Calculate Bayesian Posterior, extremely slow!"""
        assert len(torch.unique(user_ids)) == 1, "invalid usage"

        user_id = user_ids[0]
        params = self._model_params.copy()
        params["batch_size"] = None
        del params["interactions"]
        params["user_id"] = user_id

        predictive = Predictive(
            self._pred_model,
            guide=self._pred_guide,
            num_samples=self._n_samples,
            return_sites=[lda.Site.interactions],
            parallel=False,
        )
        items = predictive(**params)["interactions"].squeeze(1)
        counts = torch.bincount(items, minlength=self._n_items)
        # break ties by randomly adding values from [0, 1)
        counts = counts + torch.rand(counts.shape)

        return counts[item_ids]

    def _predict_point(self, user_ids, item_ids):
        """Calculate point estimate the frequentist way, fast!"""
        assert len(torch.unique(user_ids)) == 1, "invalid usage"

        user_topics = self.user_topics[user_ids]
        topic_items = self.topic_items[:, item_ids].T
        item_pops = self.pops[item_ids].unsqueeze(1)
        user_pop_devs = self.user_pop_devs[user_ids].unsqueeze(1)
        topic_prefs = topic_items + torch.exp(user_pop_devs) * item_pops

        dot = user_topics * topic_prefs
        if dot.dim() > 1:  # handles case where embedding_dim=1
            dot = dot.sum(1)

        return dot

    def _predict(self, user_ids, item_ids):
        if self.predict_posterior:
            return self._predict_posterior(user_ids, item_ids)
        else:
            return self._predict_point(user_ids, item_ids)

    def save(self, filename):
        pyro.get_param_store().save(filename)
        return self

    def load(self, filename, interactions):
        """Load model, interactions are only used to infer metadata"""
        self._n_iter = 0
        pyro.get_param_store().load(filename)
        self.fit(interactions, clear_params=False)
        return self


class HierLDA4RecEst(LDA4RecEst):
    def __init__(self, **kwargs):
        super().__init__(model=lda.hier_model, guide=lda.hier_guide, **kwargs)


class HierVarLDA4RecEst(LDA4RecEst):
    def __init__(self, **kwargs):
        super().__init__(model=lda.hier_model, guide=lda.hier_var_guide, **kwargs)


class BaseEstimator(EstimatorMixin, metaclass=ABCMeta):
    """Base estimator handling implicit feedback training and prediction"""

    def __init__(
        self,
        *,
        model_class,
        loss,
        embedding_dim,
        n_iter=10,
        batch_size=128,
        l2=0.0,
        learning_rate=1e-2,
        optimizer=None,
        use_cuda=False,
        rng=None,
        sparse=False,
        deque_max_len=10,
        conv_slope_max=-1e-4,
    ):
        self._model_class = model_class
        self._embedding_dim = embedding_dim
        self._use_cuda = use_cuda
        self._rng = np.random.default_rng(rng)
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._optimizer = optimizer
        self._sparse = sparse
        self._conv_deque = deque(maxlen=deque_max_len)
        self._conv_slope_max = conv_slope_max
        set_seed(self._rng.integers(0, 2 ** 32 - 1), cuda=self._use_cuda)

        self._loss = {
            "bpr": bpr_loss,
            "logistic": logistic_loss,
            "hinge": hinge_loss,
            "adpative-hinge": adaptive_hinge_loss,
        }[loss]

    def _initialize(self, interactions):
        self._n_users = interactions.n_users
        self._n_items = interactions.n_items

        model = self._model_class(
            self._n_users,
            self._n_items,
            embedding_dim=self._embedding_dim,
            sparse=self._sparse,
        )
        self._model = gpu(model, self._use_cuda)

        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self._model.parameters(), weight_decay=self._l2, lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer(self._model.parameters())

    def converged(self, loss):
        self._conv_deque.append(loss)
        if not len(self._conv_deque) == self._conv_deque.maxlen:
            return False

        diffs = np.diff(self._conv_deque)
        return np.mean(diffs) >= self._conv_slope_max

    def fit(self, interactions: Interactions):
        """Fit the model"""
        run = neptune.get_last_run()
        self._initialize(interactions)
        self._model.train(True)
        self._conv_deque.clear()

        epoch_loss = None

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        for epoch_num in range(self._n_iter):

            users, items = shuffle(user_ids, item_ids, rng=self._rng)

            user_ids_tensor = gpu(torch.from_numpy(users), self._use_cuda)
            item_ids_tensor = gpu(torch.from_numpy(items), self._use_cuda)

            epoch_loss = 0.0
            minibatch_num = -1
            batches = minibatch(
                user_ids_tensor, item_ids_tensor, batch_size=self._batch_size
            )
            for minibatch_num, (batch_user, batch_item) in enumerate(batches):
                positive_prediction = self._model(batch_user, batch_item)
                negative_prediction = self._get_negative_prediction(batch_user)

                self._optimizer.zero_grad()

                loss = self._loss(positive_prediction, negative_prediction)
                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            if minibatch_num == -1:
                raise RuntimeError("There is not even a single mini-batch to train on!")

            epoch_loss /= minibatch_num + 1

            run["train/loss"].log(epoch_loss)
            _logger.info("Epoch {: >5d}: loss {}".format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError("Degenerate epoch loss: {}".format(epoch_loss))

            if self.converged(epoch_loss):
                _logger.info("Converged after {} epochs.".format(epoch_num))
                break

        run["train/n_epochs"].log(epoch_num)
        return epoch_loss

    # ToDo: Check if we cannot just do everything here in PyTorch
    def _get_negative_prediction(self, user_ids):
        """Uniformly samples negative items from the whole item set"""
        negative_items = sample_items(self._model.n_items, len(user_ids), rng=self._rng)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)
        negative_prediction = self._model(user_ids, negative_var)

        return negative_prediction

    def save(self, filename):
        torch.save(self._model.state_dict(), filename)
        return self

    def load(self, filename, interactions):
        """Load model, interactions are only used to infer metadata"""
        self._initialize(interactions)
        self._model.load_state_dict(torch.load(filename))
        self._model.eval()
        return self


class LDATrafoMixin(metaclass=ABCMeta):
    """Mixin transforming MF to adjoint LDA formulation"""

    def __init__(self, *args, **kwargs):
        # Toggle this instance variable for predicting with LDA formulation
        self.lda_trafo = False
        super().__init__(*args, **kwargs)

    def get_nmf_params(self, user_id=None):
        """Get NMF representation for a single user_id or all (if None)

        From the lemma of the paper
        """
        if user_id is None:
            user_ids = torch.arange(self._n_users, dtype=torch.int64)
        else:
            user_ids = user_id.expand(1)
        item_ids = torch.arange(self._n_items, dtype=torch.int64)

        w = self._model.user_embeddings(user_ids).detach()
        b = self._model.item_biases(item_ids).squeeze().detach()
        h = self._model.item_embeddings(item_ids).detach()

        w_pos, w_neg = torch.zeros_like(w), torch.zeros_like(w)
        pos_mask = w >= 0
        neg_mask = ~pos_mask
        w_pos[pos_mask], w_neg[neg_mask] = w[pos_mask], -w[neg_mask]
        w = torch.cat([w_pos, w_neg], dim=1)
        h = torch.cat([h, -h], dim=1)

        # choose the minimal offset to guarantee non-negativity, not just max of abs.
        h += torch.minimum(torch.min(h, dim=0).values, torch.zeros(h.shape[1])).abs()
        b += torch.minimum(torch.min(b), torch.zeros(1)).abs()

        assert torch.all(h >= 0.0)
        assert torch.all(w >= 0.0)
        assert torch.all(b >= 0.0)

        return w, h, b

    def get_lda_params(self, user_id=None, eps=1e-6):
        """Get adjoint LDA formulation of a single user_id or all (if None)

        From the theorem of the paper
        """
        w, h, b = self.get_nmf_params(user_id)
        n_users, n_topics = w.shape

        t = w.sum(dim=1)
        h_sum = h.sum(dim=0, keepdim=True).expand(n_users, -1)
        bt_sum = (b.sum(dim=0) / t).unsqueeze(-1).expand(-1, n_topics)
        n = h_sum + bt_sum
        v = w * n
        v = v / v.sum(dim=1, keepdim=True)

        assert torch.all(v >= 0.0)
        assert torch.all((v.sum(dim=1) - 1.0).abs() <= eps)

        # corresponding naming from adjoint LDA problem
        # delta = b
        # phi = h
        # theta = v
        # lambda = 1/t

        return v, t, h, b

    def get_item_probs(self, user_id, eps=1e-6) -> np.array:
        v, t, h, b = self.get_lda_params(user_id, eps=eps)
        # from the proof of the theorem
        g = h + b.unsqueeze(-1) / t
        n = g.sum(dim=0)
        g = g / n

        assert torch.all(g >= 0.0)
        topic_sums = (g.sum(dim=0) - np.ones(g.shape[1])).abs()
        assert torch.all(topic_sums <= eps * topic_sums.shape[0])

        probs = torch.matmul(v, g.T).squeeze()

        assert torch.all(probs >= 0.0)
        assert (probs.sum() - 1.0).abs() <= eps

        return probs

    def _predict(self, user_ids, item_ids):
        if self.lda_trafo:
            assert len(torch.unique(user_ids)) == 1, "invalid usage"
            return self.get_item_probs(user_ids[0], eps=1e-4)
        else:
            # dispatch to BaseEstimator, if the gods of MRO have mercy on me
            return super()._predict(user_ids, item_ids)


class MFEst(LDATrafoMixin, BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=MFNet, loss=loss, **kwargs)


class SNMFEst(LDATrafoMixin, BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=SNMFNet, loss=loss, **kwargs)


class NMFEst(LDATrafoMixin, BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=NMFNet, loss=loss, **kwargs)
