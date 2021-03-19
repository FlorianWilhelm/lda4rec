# -*- coding: utf-8 -*-
"""
Estimators

Note: The Implicit Est class is more or less FMModel from Spotlight
"""
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn.functional as F
import torch.optim as optim

from .lda import Param
from .losses import adaptive_hinge_loss, bpr_loss, hinge_loss, logistic_loss
from .nets import BilinearNet, PosBilinearNet
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

        Parameters
        ----------

        user_ids: int or array
           If int, will predict the recommendation scores for this
           user for all items in item_ids. If an array, will predict
           scores for all (user, item) pairs defined by user_ids and
           item_ids.
        item_ids: array, optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.
        cartesian: bool, optional
            Calculate the prediction for each item times each user

        Returns
        -------

        predictions: np.array
            Predicted scores for all items in item_ids.
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

    def __init__(self):
        self.pops = None
        self._n_users = None
        self._n_items = None
        self._use_cuda = False

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


class LDA4RecEst(EstimatorMixin):
    """Estimator using LDA"""

    def __init__(self, *, model=None, params=None, n_items, **kwargs):
        # for compatibility with BaseEstimator
        super().__init__()
        params = pyro.get_param_store() if params is None else params
        self.params = params
        self.pops = params[Param.item_pops_loc].detach()
        self.topic_items = params[Param.topic_items_loc].detach()
        self.user_topics = F.softmax(params[Param.user_topics_logits], dim=-1).detach()
        # ToDo: Do this in fit!!! Hack
        self._model.n_items = n_items

    def fit(self, interactions):
        pass

    def _predict(self, user_ids, item_ids):
        user_embedding = self.user_topics[user_ids]
        item_embedding = self.topic_items[:, item_ids].T

        dot = user_embedding * item_embedding
        if dot.dim() > 1:  # handles case where embedding_dim=1
            dot = dot.sum(1)

        return dot + self.pops[item_ids]


class BaseEstimator(EstimatorMixin, metaclass=ABCMeta):
    """Factorization Machine estimator for implicit feedback using BPR"""

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
        random_state=None,
        sparse=False,
    ):
        self._model_class = model_class
        self._embedding_dim = embedding_dim
        self._use_cuda = use_cuda
        self._random_state = random_state or np.random.RandomState()
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._optimizer = optimizer
        self._sparse = sparse
        set_seed(self._random_state.randint(-(10 ** 8), 10 ** 8), cuda=self._use_cuda)

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

    def fit(self, interactions):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.Interactions`
            The input dataset.
        """
        self._initialize(interactions)
        self._model.train(True)

        epoch_loss = None

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        for epoch_num in range(self._n_iter):

            users, items = shuffle(user_ids, item_ids, random_state=self._random_state)

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

            _logger.info("Epoch {}: loss {}".format(epoch_num, epoch_loss))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError("Degenerate epoch loss: {}".format(epoch_loss))

        return epoch_loss

    # ToDo: Check if we cannot just do everything here in PyTorch
    def _get_negative_prediction(self, user_ids):
        """Uniformly samples negative items from the whole item set"""
        negative_items = sample_items(
            self._model.n_items, len(user_ids), random_state=self._random_state
        )
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)
        negative_prediction = self._model(user_ids, negative_var)

        return negative_prediction


class BilinearEst(BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=BilinearNet, loss=loss, **kwargs)


class PosBilinearEst(BaseEstimator):
    def __init__(self, *, loss="bpr", **kwargs):
        super().__init__(model_class=PosBilinearNet, loss=loss, **kwargs)
