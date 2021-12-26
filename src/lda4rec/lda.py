""""
Latent Dirichlet Allocation Model for Collaborative Filtering

Shape notation: (batch_shape | event_shape), e.g. (10, 2 | 3)

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
from dataclasses import dataclass
from typing import Callable, List, Optional

import pyro
import pyro.distributions as dist
import pyro.optim
import torch
import torch.nn.functional as F
from pyro import poutine

from .utils import reparam_beta

# missing value for int
NA = -999

# Type declaration
Model = Callable[..., torch.Tensor]
Guide = Callable[..., None]


class Plate:
    topics = "plate_topics"
    users = "plate_users"
    interactions = "plate_interactions"


class Site:
    user_topics_weights = "user_topics_weights"
    interactions = "interactions"
    topic_items = "topic_items"
    item_pops = "item_pops"
    user_topics = "user_topics"
    user_inv = "user_inv"
    item_topics = "item_topics"
    user_pop_devs = "user_pop_devs"
    topic_prior = "topic_prior"
    user_pop_devs_prior_mu = "user_pop_devs_prior_mu"
    user_pop_devs_prior_sigma = "user_pop_devs_prior_sigma"

    @classmethod
    def all(cls) -> List[str]:
        return [
            a for a in dir(cls) if not (a.startswith("_") or callable(getattr(cls, a)))
        ]


class Param:
    topic_items_loc = "topic_items_loc"
    topic_items_scale = "topic_items_scale"
    user_topics_logits = "user_topics_logits"
    user_inv_logits = "user_inv_logits"
    item_pops_loc = "item_pops_loc"
    item_pops_scale = "item_pops_scale"
    user_pop_devs_loc = "user_pop_devs_loc"
    user_pop_devs_scale = "user_pop_devs_scale"
    topic_prior_logits = "topic_prior_logits"
    topic_prior_alpha = "topic_prior_alpha"
    topic_prior_beta = "topic_prior_beta"
    topic_prior_p = "topic_prior_p"
    topic_prior_q = "topic_prior_q"
    user_pop_devs_prior_mu_loc = "user_pop_devs_prior_mu_loc"
    user_pop_devs_prior_mu_scale = "user_pop_devs_prior_mu_scale"
    user_pop_devs_prior_sigma_loc = "user_pop_devs_prior_sigma_loc"
    user_pop_devs_prior_sigma_scale = "user_pop_devs_prior_sigma_scale"


@dataclass()
class ModelData:
    interactions: torch.Tensor
    item_pops: torch.Tensor
    user_topics: torch.Tensor


def model(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> ModelData:
    """Bayesian type model

    Args:
        interactions: 2-d array of shape (n_interactions, n_users)
    """
    alpha = 1.0 / n_topics if alpha is None else alpha
    n_interactions = interactions.shape[0]

    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)

    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )

    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None:
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            interactions = interactions[:, ind]

        user_topics = pyro.sample(  # (n_users | n_topics)
            Site.user_topics,
            dist.Dirichlet(alpha * torch.ones(n_topics)),  # prefer sparse
        )

        user_pop_devs = pyro.sample(  # (n_users | )
            Site.user_pop_devs,
            dist.LogNormal(-0.5 * torch.ones(1), 0.5),
        ).unsqueeze(1)

        with pyro.plate(Plate.interactions, n_interactions):
            item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                # final preference depends on the topic distribution,
                # the item popularity and how much a user cares about
                # item popularity
                prefs = topic_items[item_topics] + user_pop_devs * item_pops
                interactions = pyro.sample(  # (n_interactions, n_users)
                    Site.interactions,
                    dist.Categorical(logits=prefs),
                    obs=interactions,
                )

    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
    )


def hier_model(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> ModelData:
    """Hierarchical Bayesian type model

    Args:
        interactions: 2-d array of shape (n_interactions, n_users)
    """
    alpha = 1.0 / n_topics if alpha is None else alpha
    n_interactions = interactions.shape[0]

    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)

    beta_alpha, beta_beta = reparam_beta(alpha, 0.9)
    topic_prior = pyro.sample(  # ( | n_topics)
        Site.topic_prior,
        dist.Beta(beta_alpha * torch.ones(n_topics), beta_beta).to_event(1),
    ).unsqueeze(0)

    user_pop_devs_prior_mu = pyro.sample(  # ( | 1)
        Site.user_pop_devs_prior_mu, dist.Normal(-0.5 * torch.ones(1), 1.0)
    )

    user_pop_devs_prior_sigma = pyro.sample(  # ( | 1)
        Site.user_pop_devs_prior_sigma, dist.HalfNormal(scale=4.0 * torch.ones(1))
    )

    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )

    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None:
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            interactions = interactions[:, ind]

        user_topics = pyro.sample(  # (n_users | n_topics)
            Site.user_topics,
            dist.Dirichlet(topic_prior),
        )

        user_pop_devs = pyro.sample(  # (n_users | )
            Site.user_pop_devs,
            dist.LogNormal(user_pop_devs_prior_mu, user_pop_devs_prior_sigma),
        ).unsqueeze(1)

        with pyro.plate(Plate.interactions, n_interactions):
            item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                # final preference depends on the topic distribution,
                # the item popularity and how much a user cares about
                # item popularity
                prefs = topic_items[item_topics] + user_pop_devs * item_pops
                interactions = pyro.sample(  # (n_interactions, n_users)
                    Site.interactions,
                    dist.Categorical(logits=prefs),
                    obs=interactions,
                )

    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
    )


def guide(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.normal(torch.zeros(n_users, n_topics), 1.0 / n_topics),
    )

    user_pop_devs = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )

    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        pyro.sample(
            Site.user_pop_devs,
            dist.Delta(torch.exp(user_pop_devs[ind])),
        )

        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )


def hier_guide(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    alpha = 1.0 / n_topics if alpha is None else alpha

    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    init_alpha, init_beta = reparam_beta(alpha, 0.1)
    topic_prior_alpha = pyro.param(
        Param.topic_prior_alpha,
        lambda: init_alpha * torch.ones(n_topics),
        constraint=dist.constraints.interval(0.05, 1.0),
    )
    topic_prior_beta = pyro.param(
        Param.topic_prior_beta,
        lambda: init_beta * torch.ones(n_topics),
        constraint=dist.constraints.interval(0.05, 1.0),
    )
    pyro.sample(
        Site.topic_prior, dist.Beta(topic_prior_alpha, topic_prior_beta).to_event(1)
    )

    user_pop_devs_prior_mu_loc = pyro.param(
        Param.user_pop_devs_prior_mu_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_mu_scale = pyro.param(
        Param.user_pop_devs_prior_mu_scale,
        lambda: torch.normal(mean=0.5 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        Site.user_pop_devs_prior_mu,
        dist.Normal(user_pop_devs_prior_mu_loc, user_pop_devs_prior_mu_scale),
    )

    user_pop_devs_prior_sigma_loc = pyro.param(
        Param.user_pop_devs_prior_sigma_loc,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_sigma_scale = pyro.param(
        Param.user_pop_devs_prior_sigma_scale,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        Site.user_pop_devs_prior_sigma,
        dist.TransformedDistribution(
            dist.Normal(
                loc=user_pop_devs_prior_sigma_loc, scale=user_pop_devs_prior_sigma_scale
            ),
            transforms=dist.transforms.ExpTransform(),
        ),
    )

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.normal(torch.zeros(n_users, n_topics), 1.0 / n_topics),
    )
    user_pop_devs = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )
    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        pyro.sample(
            Site.user_pop_devs,
            dist.Delta(torch.exp(user_pop_devs[ind])),
        )

        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )


def hier_var_guide(
    interactions: torch.Tensor,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    alpha = 1.0 / n_topics if alpha is None else alpha

    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    topic_prior_p = pyro.param(
        Param.topic_prior_p,
        lambda: torch.ones(1),
        constraint=dist.constraints.interval(0.1, 1.0),
    )
    topic_prior_q = pyro.param(
        Param.topic_prior_q,
        lambda: 0.9 * torch.ones(1),
        constraint=dist.constraints.interval(0.2, 1.0),
    )

    pyro.sample(
        Site.topic_prior,
        dist.Delta(
            (topic_prior_p.log() + topic_prior_q.log() * torch.arange(n_topics)).exp()
        ).to_event(1),
    )

    user_pop_devs_prior_mu_loc = pyro.param(
        Param.user_pop_devs_prior_mu_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_mu_scale = pyro.param(
        Param.user_pop_devs_prior_mu_scale,
        lambda: torch.normal(mean=0.5 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        Site.user_pop_devs_prior_mu,
        dist.Normal(user_pop_devs_prior_mu_loc, user_pop_devs_prior_mu_scale),
    )

    user_pop_devs_prior_sigma_loc = pyro.param(
        Param.user_pop_devs_prior_sigma_loc,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.5),
    )
    user_pop_devs_prior_sigma_scale = pyro.param(
        Param.user_pop_devs_prior_sigma_scale,
        lambda: torch.normal(mean=1.0 * torch.ones(1), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(
        Site.user_pop_devs_prior_sigma,
        dist.TransformedDistribution(
            dist.Normal(
                loc=user_pop_devs_prior_sigma_loc, scale=user_pop_devs_prior_sigma_scale
            ),
            transforms=dist.transforms.ExpTransform(),
        ),
    )

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.normal(torch.zeros(n_users, n_topics), 1.0 / n_topics),
    )
    user_pop_devs = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )
    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        pyro.sample(
            Site.user_pop_devs,
            dist.Delta(torch.exp(user_pop_devs[ind])),
        )

        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )


def pred_model(
    user_id: int,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    """Like model but for prediction

    Args:
        interactions: 2-d array of shape (n_interactions, n_users)
    """
    alpha = 1.0 / n_topics if alpha is None else alpha

    # omega
    item_pops = pyro.sample(  # ( | n_items)
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    ).unsqueeze(0)

    with pyro.plate(Plate.topics, n_topics):
        topic_items = pyro.sample(  # (n_topics | n_items)
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )

    user_topics = pyro.sample(  # (n_users | n_topics)
        Site.user_topics,
        dist.Dirichlet(alpha * torch.ones(n_topics)),  # prefer sparse
    )

    user_pop_devs = pyro.sample(  # (n_users | )
        Site.user_pop_devs,
        dist.LogNormal(-0.5 * torch.ones(1), 0.5),
    ).unsqueeze(1)

    item_topics = pyro.sample(  # (n_ratings_per_user | n_users)
        Site.item_topics,
        dist.Categorical(user_topics),
        infer={"enumerate": "parallel"},
    )

    # final preference depends on the topic distribution,
    # the item popularity and how much a user cares about
    # item popularity
    prefs = topic_items[item_topics] + user_pop_devs * item_pops
    interactions = pyro.sample(  # (n_interactions, n_users)
        Site.interactions,
        dist.Categorical(logits=prefs),
    )

    return interactions


def pred_guide(
    user_id: int,
    *,
    n_topics: int,
    n_users: int,
    n_items: int,
    alpha: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    item_pops_loc = pyro.param(
        Param.item_pops_loc,
        lambda: torch.normal(mean=torch.zeros(n_items), std=1.0),
    )
    item_pops_scale = pyro.param(
        Param.item_pops_scale,
        lambda: torch.normal(mean=torch.ones(n_items), std=0.5).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )
    pyro.sample(Site.item_pops, dist.Normal(item_pops_loc, item_pops_scale).to_event(1))

    topic_items_loc = pyro.param(
        Param.topic_items_loc,
        lambda: torch.normal(mean=torch.zeros(n_topics, n_items), std=0.5),
    )
    topic_items_scale = pyro.param(
        Param.topic_items_scale,
        lambda: torch.normal(mean=torch.ones(n_topics, n_items), std=0.5).clamp(
            min=0.1
        ),
        constraint=dist.constraints.positive,
    )
    with pyro.plate(Plate.topics, n_topics):
        pyro.sample(
            Site.topic_items,
            dist.Normal(topic_items_loc, topic_items_scale).to_event(1),
        )

    user_topics_logits = pyro.param(
        Param.user_topics_logits,
        lambda: torch.zeros(n_users, n_topics),
    )
    user_pop_devs_loc = pyro.param(
        Param.user_pop_devs_loc,
        lambda: torch.normal(mean=-0.5 * torch.ones(n_users), std=0.1),
    )
    user_pop_devs_scale = pyro.param(
        Param.user_pop_devs_scale,
        lambda: torch.normal(mean=0.5 * torch.ones(n_users), std=0.1).clamp(min=0.1),
        constraint=dist.constraints.positive,
    )

    pyro.sample(
        Site.user_pop_devs,
        dist.LogNormal(
            loc=user_pop_devs_loc[user_id], scale=user_pop_devs_scale[user_id]
        ),
    )

    # use Delta dist for MAP avoiding high variances with Dirichlet posterior
    pyro.sample(
        Site.user_topics,
        dist.Delta(F.softmax(user_topics_logits[user_id], dim=-1), event_dim=1),
    )
