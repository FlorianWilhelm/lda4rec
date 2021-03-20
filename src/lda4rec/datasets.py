"""
Different data sets

Note: Some parts copied over from Spotlight (MIT)
"""
from __future__ import annotations

import logging
import os
import shutil
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Iterable, Optional, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import scipy.sparse as sparse
from tqdm import tqdm

_logger = logging.getLogger(__name__)

Resource = namedtuple("Resource", ["url", "path", "interactions", "read_csv_args"])

# all available datasets
MOVIELENS_20M = Resource(
    path="ml-20m/raw.zip",
    interactions="ratings.csv",
    read_csv_args={"names": ["user_id", "item_id", "rating", "timestamp"], "header": 0},
    url="http://files.grouplens.org/datasets/movielens/ml-20m.zip",
)
# ToDo: Add Movielens 1m
# hier alles implementieren was es bei lightfm gibt
MOVIELENS_100K_OLD = Resource(
    path="ml100k-old/raw.zip",
    interactions="u.data",
    read_csv_args={
        "names": ["user_id", "item_id", "rating", "timestamp"],
        "header": None,
        "sep": "\t",
    },
    url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
)
MOVIELENS_100K = Resource(
    path="ml-latest-small/raw.zip",
    interactions="ratings.csv",
    read_csv_args={"names": ["user_id", "item_id", "rating", "timestamp"], "header": 0},
    url="http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
)

DATA_DIR = os.path.join(os.path.expanduser("~"), ".lda4rec")


def compact(
    iterable: Iterable, return_map: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Applies Compacter to get consecutive elements, e.g.
    [1, 2, 5, 2] or ['a', 'b', 'e', 'b'] become
    a compact, consecutive integer representation [0, 1, 2, 1]
    """
    mapper = defaultdict()
    mapper.default_factory = mapper.__len__
    new_order = np.array([mapper[elem] for elem in iterable], dtype=np.int32)
    if return_map:
        inv_mapper = {v: k for k, v in mapper.items()}
        arg_map = np.array([inv_mapper[i] for i in range(len(inv_mapper))])
        return new_order, arg_map
    else:
        return new_order


class UnknownDataset(ValueError):
    pass


def get_dataset(name: str, *args, **kwargs) -> Interactions:
    loader = DataLoader(*args, **kwargs)
    if name.lower() == "movielens-100k":
        return loader.load_movielens("100k")
    else:
        raise UnknownDataset(f"Unknown dataset: {name}")


class DataLoader(object):
    def __init__(self, data_dir=DATA_DIR, show_progress=True):
        self.data_dir = data_dir
        self.show_progress = show_progress

    def get_data(self, resource, download_if_missing=True):
        dest_path = os.path.join(os.path.abspath(self.data_dir), resource.path)
        dir_path = os.path.dirname(dest_path)
        create_data_dir(dir_path)

        if not os.path.isfile(dest_path):
            if download_if_missing:
                download(resource.url, dest_path, self.show_progress)
            else:
                raise IOError("Dataset missing.")
        if count_files(dir_path) == 1:
            unzip_flat(dest_path, dir_path)

        return dir_path

    def load_movielens(self, variant="100k"):
        variants = {
            "100k": MOVIELENS_100K,
            "100k-old": MOVIELENS_100K_OLD,
            "20m": MOVIELENS_20M,
        }
        resource = variants[variant]
        dir_path = self.get_data(resource)
        ratings = os.path.join(dir_path, resource.interactions)
        df = pd.read_csv(ratings, **resource.read_csv_args)
        user_ids = df.user_id.values
        item_ids = df.item_id.values
        ratings = df.rating.values
        user_ids = compact(user_ids)
        item_ids = compact(item_ids)
        interactions = Interactions(
            user_ids=user_ids, item_ids=item_ids, ratings=ratings
        )
        return interactions


def download(url, dest_path, show_progress=True):
    req = requests.get(url, stream=True)
    req.raise_for_status()

    chunk_size = 2 ** 20  # Megabyte, compare with unit below!
    bytestream = req.iter_content(chunk_size=chunk_size)
    if show_progress:
        file_size = int(req.headers["Content-Length"])
        n_bars = file_size // chunk_size

        bytestream = tqdm(
            bytestream, unit="MB", total=n_bars, ascii=True, desc=dest_path
        )

    with open(dest_path, "wb") as fd:
        for chunk in bytestream:
            fd.write(chunk)


def unzip_flat(src_path, dest_dir):
    """Unzip all files in archive discarding the directory structure"""
    with ZipFile(src_path) as zip_file:
        for obj in zip_file.filelist:
            if obj.is_dir():
                continue
            source = zip_file.open(obj.filename)
            filename = os.path.basename(obj.filename)
            with open(os.path.join(dest_dir, filename), "wb") as target:
                shutil.copyfileobj(source, target)


def create_data_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def count_files(path):
    return len(
        [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    )


class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions, but can also be enriched with ratings, timestamps,
    and interaction weights.

    For *implicit feedback* scenarios, user ids and item ids should
    only be provided for user-item pairs where an interaction was
    observed. All pairs that are not provided are treated as missing
    observations, and often interpreted as (implicit) negative
    signals.

    For *explicit feedback* scenarios, user ids, item ids, and
    ratings should be provided for all user-item-rating triplets
    that were observed in the dataset.

    Parameters
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    num_users: int, optional
        Number of distinct users in the dataset.
        Must be larger than the maximum user id
        in user_ids.
    num_items: int, optional
        Number of distinct items in the dataset.
        Must be larger than the maximum item id
        in item_ids.

    Attributes
    ----------

    user_ids: array of np.int32
        array of user ids of the user-item pairs
    item_ids: array of np.int32
        array of item ids of the user-item pairs
    ratings: array of np.float32, optional
        array of ratings
    timestamps: array of np.int32, optional
        array of timestamps
    weights: array of np.float32, optional
        array of weights
    n_users: int, optional
        Number of distinct users in the dataset.
    n_items: int, optional
        Number of distinct items in the dataset.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        ratings: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
    ):

        self.n_users = n_users or int(user_ids.max() + 1)
        self.n_items = n_items or int(item_ids.max() + 1)

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings if ratings is not None else np.ones(user_ids.shape)
        self.timestamps = timestamps
        self.weights = weights

        self._check()

    def __repr__(self) -> str:
        return (
            "<Interactions dataset ({n_users} users x {n_items} items "
            "x {n_interactions} interactions)>".format(
                n_users=self.n_users, n_items=self.n_items, n_interactions=len(self)
            )
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: Union[slice, int]) -> Interactions:
        return Interactions(
            self.user_ids[idx],
            self.item_ids[idx],
            ratings=self.ratings[idx],
            timestamps=self.timestamps[idx] if self.timestamps is not None else None,
            weights=self.weights[idx] if self.weights is not None else None,
            n_users=self.n_users,
            n_items=self.n_items,
        )

    def _check(self):
        if self.user_ids.max() >= self.n_users:
            raise ValueError(
                "Maximum user id greater " "than declared number of users."
            )
        if self.item_ids.max() >= self.n_items:
            raise ValueError(
                "Maximum item id greater " "than declared number of items."
            )

        n_interactions = len(self.user_ids)

        for name, value in (
            ("item IDs", self.item_ids),
            ("ratings", self.ratings),
            ("timestamps", self.timestamps),
            ("weights", self.weights),
        ):

            if value is None:
                continue

            if len(value) != n_interactions:
                raise ValueError(
                    "Invalid {} dimensions: length "
                    "must be equal to number of interactions".format(name)
                )

    def select_by_mask_(self, mask: np.ndarray):
        self.user_ids = self.user_ids[mask]
        self.item_ids = self.item_ids[mask]
        self.ratings = self.ratings[mask]
        if self.weights is not None:
            self.weights = self.weights[mask]
        if self.timestamps is not None:
            self.timestamps = self.timestamps[mask]

    def remove_user_ids_(self, user_ids: np.ndarray):
        mask = ~np.in1d(self.user_ids, user_ids)
        self.select_by_mask_(mask)

    def remove_item_ids_(self, item_ids: np.ndarray):
        mask = ~np.in1d(self.item_ids, item_ids)
        self.select_by_mask_(mask)

    def max_user_interactions_(self, n: int, rng=None):
        rng = np.random.default_rng(rng)
        exceed = pd.Series(self.user_ids).value_counts()
        user_ids = exceed.loc[exceed - n > 0].index.to_numpy()

        mat = self.to_csr()
        for user_id in user_ids:
            n_removals = len(mat[user_id].indices) - n
            item_ids = rng.choice(mat[user_id].indices, size=n_removals, replace=False)
            mat[user_id, item_ids] = 0

        mat.eliminate_zeros()
        mat = mat.tocoo()
        self.user_ids = mat.row
        self.item_ids = mat.col
        self.ratings = mat.data

    def implicit_(self, pivot: float):
        """Makes the current dataset implicit by setting values < pivot to 0
        and values >= pivot to 1"""
        assert np.max(self.ratings) >= pivot
        mask = self.ratings >= pivot
        self.ratings[mask] = 1
        self.ratings = self.ratings[mask]
        self.user_ids = self.user_ids[mask]
        self.item_ids = self.item_ids[mask]

    def implicit(self, pivot: float) -> Interactions:
        clone = self.copy()
        clone.implicit_(pivot)
        return clone

    def binarize_(self, pivot: float):
        """Makes the current dataset implicit by setting values < pivot to -1
        and values >= pivot to 1"""
        assert np.max(self.ratings) >= pivot
        mask = self.ratings >= pivot
        self.ratings[mask] = 1
        self.ratings[~mask] = -1

    def binarize(self, pivot: float) -> Interactions:
        clone = self.copy()
        clone.binarize_(pivot)
        return clone

    def compact_(self) -> Tuple[np.ndarray, np.ndarray]:
        self.user_ids, user_map = compact(self.user_ids, return_map=True)
        self.item_ids, item_map = compact(self.item_ids, return_map=True)
        return user_map, item_map

    def copy(self) -> Interactions:
        return deepcopy(self)

    def to_coo(self) -> sparse.coo_matrix:
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = self.ratings

        return sparse.coo_matrix((data, (row, col)), shape=(self.n_users, self.n_items))

    def to_csr(self) -> sparse.csr_matrix:
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.to_coo().tocsr()

    def to_pandas(self) -> pd.DataFrame:
        """
        Transform to Pandas DataFrame
        """
        df = pd.DataFrame(
            data={
                "user_id": pd.Series(self.user_ids, dtype=np.int32),
                "item_id": pd.Series(self.item_ids, dtype=np.int32),
                "rating": pd.Series(self.ratings, dtype=np.float),
            }
        )
        return df

    def to_numpy(self) -> np.ndarray:
        """
        Transform to float numpy array
        """
        return np.vstack([self.user_ids, self.item_ids, self.ratings]).T

    def to_ratings_per_user(self, n_interactions: Optional[int] = None) -> np.ndarray:
        df = (
            self.to_pandas()
            .drop("rating", axis=1)
            .assign(idx=lambda df: df.groupby(["user_id"]).cumcount())
            .set_index(keys=["user_id", "idx"])
        )
        idx_dims = [level.max() + 1 for level in df.index.levels]

        if n_interactions is not None:
            idx_dims[-1] = n_interactions

        cube = np.empty(idx_dims, np.int)
        cube.fill(-999)
        cube[tuple(np.array(df.index.to_list()).T)] = df.values[:, 0]
        return cube.T


class InteractionsFactory(object):
    def __init__(self):
        self.user_ids = list()
        self.item_ids = list()
        self.ratings = list()

    def add_interaction(self, user_id: int, item_id: int, rating: float):
        self.user_ids.append(user_id)
        self.item_ids.append(item_id)
        self.ratings.append(rating)

    def __len__(self):
        return len(self.ratings)

    def make_interactions(self) -> Interactions:
        user_ids = np.array(self.user_ids, dtype=np.int)
        item_ids = np.array(self.item_ids, dtype=np.int)
        ratings = np.array(self.ratings, dtype=float)
        return Interactions(user_ids, item_ids, ratings)


def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]


def shuffle_interactions(interactions: Interactions, rng=None):
    """Shuffle interactions randomly"""
    rng = np.random.default_rng(rng)

    shuffle_indices = np.arange(len(interactions.user_ids))
    rng.shuffle(shuffle_indices)

    return Interactions(
        interactions.user_ids[shuffle_indices],
        interactions.item_ids[shuffle_indices],
        ratings=_index_or_none(interactions.ratings, shuffle_indices),
        timestamps=_index_or_none(interactions.timestamps, shuffle_indices),
        weights=_index_or_none(interactions.weights, shuffle_indices),
        n_users=interactions.n_users,
        n_items=interactions.n_items,
    )


def random_train_test_split(
    interactions: Interactions, test_percentage: float = 0.2, rng=None
):
    """Randomly split interactions between training and testing"""

    interactions = shuffle_interactions(interactions, rng=rng)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(
        interactions.user_ids[train_idx],
        interactions.item_ids[train_idx],
        ratings=_index_or_none(interactions.ratings, train_idx),
        timestamps=_index_or_none(interactions.timestamps, train_idx),
        weights=_index_or_none(interactions.weights, train_idx),
        n_users=interactions.n_users,
        n_items=interactions.n_items,
    )
    test = Interactions(
        interactions.user_ids[test_idx],
        interactions.item_ids[test_idx],
        ratings=_index_or_none(interactions.ratings, test_idx),
        timestamps=_index_or_none(interactions.timestamps, test_idx),
        weights=_index_or_none(interactions.weights, test_idx),
        n_users=interactions.n_users,
        n_items=interactions.n_items,
    )

    return train, test
