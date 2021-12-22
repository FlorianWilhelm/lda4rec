"""
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
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import click
import neptune.new as neptune
import numpy as np
import yaml
from neptune.new.types import File
from neptune.utils import get_git_info

from . import __version__, estimators
from .datasets import (
    get_dataset,
    items_per_user_train_test_split,
    random_train_test_split,
)
from .evaluations import summary
from .utils import Config, log_dataset, log_summary

_logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to config file",
)
@click.option("-s", "--silent", "silent", flag_value=True, default=False)
@click.version_option(__version__)
@click.pass_context
def main(ctx, cfg_path: Path, silent: bool):
    """Experimentation tool LDA4Rec"""

    cfg = Config(Path(cfg_path), silent=silent)
    ctx.obj = cfg  # pass Config to other commands


def init_neptune(cfg):
    neptune_cfg = cfg["neptune"]
    run = neptune.init(
        name=cfg["main"]["name"],
        **neptune_cfg,
    )
    run["experiment"] = cfg["experiment"]
    # ToDo: Change this when neptune.new allows passing it in init
    git_info = vars(get_git_info(str(Path(__file__).resolve())))
    git_info["commit_date"] = str(git_info["commit_date"])
    run["git_info"] = git_info
    return run


def setup_logging(cfg):
    logging.basicConfig(
        stream=sys.stdout,
        level=cfg["main"]["log_level"],
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # set up an additional file logger ...
    pkg_logger = __name__.split(".")[0]
    log_file = cfg["main"]["log_path"] / Path(
        f"{cfg['main']['name']}_{cfg['main']['timestamp_str']}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(cfg["main"]["log_level"])
    logging.getLogger(pkg_logger).addHandler(fh)
    _logger.info(f"Configuration:\n{cfg.yaml_content}")


@main.command(name="run")
@click.pass_obj
def run_experiment(cfg: Config):
    """Run the experiment from the config"""
    run = init_neptune(cfg)
    setup_logging(cfg)
    exp_cfg = cfg["experiment"]
    model_rng = np.random.default_rng(exp_cfg["model_seed"])
    data_rng = np.random.default_rng(exp_cfg["dataset_seed"])

    dataset = get_dataset(exp_cfg["dataset"], data_dir=cfg["main"]["data_path"])
    dataset.implicit_(exp_cfg["interaction_pivot"])  # implicit feedback
    dataset.max_user_interactions_(exp_cfg["max_user_interactions"], rng=data_rng)
    dataset.min_user_interactions_(exp_cfg["min_user_interactions"])

    if exp_cfg["train_test_split"] == "random_train_test_split":
        train, rest = random_train_test_split(
            dataset, test_percentage=0.10, rng=data_rng
        )
        test, valid = random_train_test_split(rest, test_percentage=0.5, rng=data_rng)
    elif exp_cfg["train_test_split"] == "items_per_user_train_test_split":
        train, rest = items_per_user_train_test_split(
            dataset, n_items_per_user=9, rng=data_rng
        )
        test, valid = items_per_user_train_test_split(
            rest, n_items_per_user=9, rng=data_rng
        )
    else:
        raise RuntimeError(f"Unknown train-test-split: {exp_cfg['train_test_split']}")

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        log_dataset(name, data)
        _logger.info(f"{name}: {data}")
        _logger.info(f"{name}_hash: {data.hash()}")

    est_class = getattr(estimators, exp_cfg["estimator"])
    try:
        est = est_class(**exp_cfg["est_params"], rng=model_rng)
        _logger.info(f"Fitting estimator {exp_cfg['estimator']}...")
        est.fit(train)
        _logger.info("Evaluating...")
        df = summary(est, train=train, valid=valid, test=test)

        log_summary(df.reset_index())
        run["summary/df"].upload(File.as_html(df))
        _logger.info(f"Result:\n{df.reset_index()}")
    except Exception:
        run["sys/tags"].add("failure")
        raise
    else:
        run["sys/tags"].add("success")


def experiments_gen(template):
    """Generate different experiment config setups"""

    def make_configs(template, param_names, model_params_iter):
        for params in model_params_iter:
            exp_temp = deepcopy(template)
            for name, param in zip(param_names, params):
                exp_temp["experiment"]["est_params"][name] = param
            yield exp_temp

    estimators = ["LDA4RecEst", "PopEst", "MFEst"]
    datasets = ["movielens-1m", "goodbooks"]
    model_seeds = [3128845410]

    embedding_dims = [32, 48, 64, 80]
    learning_rates = [0.001]
    batch_sizes = [64, 128]
    train_test_splits = ["random_train_test_split", "items_per_user_train_test_split"]

    for estimator, model_seed, dataset, train_test_split in product(
        estimators, model_seeds, datasets, train_test_splits
    ):
        exp_temp = deepcopy(template)
        exp_cfg = exp_temp["experiment"]
        exp_cfg.update(
            {
                "dataset": dataset,
                "model_seed": model_seed,
                "estimator": estimator,
                "train_test_split": train_test_split,
            }
        )
        if estimator == "PopEst":
            exp_cfg["est_params"] = {}
            exp_temp["experiment"] = exp_cfg
            yield exp_temp
        elif estimator in ("LDA4RecEst", "HierLDA4RecEst", "HierVarLDA4RecEst"):
            params = {
                "embedding_dim": [8, 16, 32],
                "learning_rate": learning_rates,
                "batch_size": batch_sizes,
                "n_iter": [10_000],
            }
            yield from make_configs(exp_temp, params.keys(), product(*params.values()))
        elif estimator in ("MFEst", "SNMFEst", "NMFEst"):
            params = {
                "embedding_dim": embedding_dims,
                "learning_rate": learning_rates,
                "batch_size": batch_sizes,
                "n_iter": [200],
            }
            yield from make_configs(exp_temp, params.keys(), product(*params.values()))
        else:
            raise RuntimeError(f"Unknown estimator {estimator}!")


@main.command(name="create")
@click.pass_obj
def create_experiments(cfg: Config):
    """Create experiment configurations"""
    template = yaml.safe_load(cfg.yaml_content)
    for idx, experiment in enumerate(experiments_gen(template)):
        with open(cfg.path.parent / Path(f"exp_{idx}.yaml"), "w") as fh:
            yaml.dump(experiment, fh)


if __name__ == "__main__":
    main()
