""""
Latent Dirichlet Allocation Model for Collaborative Filtering

Shape notation: (batch_shape | event_shape), e.g. (10, 2 | 3)
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import pyro
import pyro.distributions as dist
import pyro.optim
import torch
import torch.nn.functional as F
from pyro import poutine

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
    ratings = "ratings"
    rating_probs = "rating_probs"
    topic_items = "topic_items"
    topic_prefs = "topic_prefs"
    item_pops = "item_pops"
    user_topics = "user_topics"
    user_inv = "user_inv"
    item_topics = "item_topics"

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


@dataclass()
class ModelData:
    interactions: torch.Tensor
    item_pops: torch.Tensor
    user_topics: torch.Tensor
    topic_prefs: torch.Tensor


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

    # omega
    item_pops = pyro.sample(
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    )  # ( | n_items)

    with pyro.plate(Plate.topics, n_topics):
        # user_topics_weights = pyro.sample(
        #     Site.user_topics_weights, dist.Gamma(alpha, 1.0)
        # )  # (n_topics)
        # beta
        topic_items = pyro.sample(
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )  # (n_topics | n_items)
        # topic preference is determined by preference but also by popularity
        topic_prefs = pyro.deterministic(
            Site.topic_prefs, topic_items + item_pops.unsqueeze(0)
        )  # (n_topics | n_items)

    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None:
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            interactions = interactions[:, ind]

        user_topics = pyro.sample(
            Site.user_topics,
            # ToDo: Try out Gamma Prior here
            dist.Dirichlet(alpha * torch.ones(n_topics)),  # prefer sparse
        )  # (n_users | n_topics)

        with pyro.plate(Plate.interactions, n_interactions):
            item_topics = pyro.sample(
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )  # (n_ratings_per_user | n_users)
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                interactions = pyro.sample(
                    Site.ratings,
                    dist.Categorical(logits=topic_prefs[item_topics]),
                    obs=interactions,
                )  # (n_interactions, n_users)

    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
        topic_prefs=topic_prefs,
    )


def model_idea(
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

    # omega
    item_pops = pyro.sample(
        Site.item_pops, dist.Normal(torch.zeros(n_items), 2.0).to_event(1)
    )  # ( | n_items)

    with pyro.plate(Plate.topics, n_topics):
        # user_topics_weights = pyro.sample(
        #     Site.user_topics_weights, dist.Gamma(alpha, 1.0)
        # )  # (n_topics)
        # beta
        topic_items = pyro.sample(
            Site.topic_items,
            dist.Normal(torch.zeros(1, n_items), 1.0).to_event(1),
        )  # (n_topics | n_items)
        # topic preference is determined by preference but also by popularity
        # topic_prefs = pyro.deterministic(
        #     Site.topic_prefs, topic_items + item_pops.unsqueeze(0)
        # )  # (n_topics | n_items)

    with pyro.plate(Plate.users, n_users) as ind:
        if interactions is not None:
            with pyro.util.ignore_jit_warnings():
                assert interactions.shape == (n_interactions, n_users)
                assert interactions.max() < n_items
            interactions = interactions[:, ind]

        user_inv = -1.0 + 2.0 * pyro.sample(
            Site.user_inv, dist.Bernoulli(0.5 * torch.ones(1))
        )  # (n_users)
        user_topics = pyro.sample(
            Site.user_topics,
            # ToDo: Try out Gamma Prior here
            dist.Dirichlet(alpha * torch.ones(n_topics)),  # prefer sparse
        )  # (n_users | n_topics)

        with pyro.plate(Plate.interactions, n_interactions):
            item_topics = pyro.sample(
                Site.item_topics,
                dist.Categorical(user_topics),
                infer={"enumerate": "parallel"},
            )  # (n_ratings_per_user | n_users)
            if interactions is not None:
                mask = interactions != NA
                interactions[~mask] = 0
            else:
                mask = True

            with poutine.mask(mask=mask):
                interactions = pyro.sample(
                    Site.ratings,
                    dist.Categorical(
                        logits=user_inv.unsqueeze(1) * topic_items[item_topics]
                        + item_pops.unsqueeze(0)
                    ),
                    obs=interactions,
                )  # (n_interactions, n_users)

    return ModelData(
        interactions=interactions,
        item_pops=item_pops,
        user_topics=user_topics,
        topic_prefs=user_topics,  # TODO: FIX ME
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
        lambda: torch.zeros(n_users, n_topics),
    )
    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )


def guide_idea(
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
        lambda: torch.zeros(n_users, n_topics),
    )
    user_inv_logits = pyro.param(
        Param.user_inv_logits,
        lambda: 2.5 * torch.ones(n_users),
    )
    with pyro.plate(Plate.users, n_users, batch_size) as ind:
        # use Delta dist for MAP avoiding high variances with Dirichlet posterior
        pyro.sample(
            Site.user_topics,
            dist.Delta(F.softmax(user_topics_logits[ind], dim=-1), event_dim=1),
        )
        pyro.sample(
            Site.user_inv,
            # use sigmoid instead of Beta avoiding high variances
            dist.Bernoulli(torch.sigmoid(user_inv_logits[ind])),
        )
