import logging
import sys
from itertools import product
from pathlib import Path

import click
import neptune
import numpy as np
import yaml
from neptune.utils import get_git_info
from neptunecontrib.api.table import log_table

from . import __version__, estimators
from .datasets import get_dataset, random_train_test_split
from .evaluations import summary
from .utils import Config, flatten_dict, log_dataset, log_summary

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
    init_cfg = neptune_cfg["init"]
    exp_cfg = neptune_cfg["create_experiment"]
    # needs to be determined explicitly because of `console_scripts`
    git_info = get_git_info(str(Path(__file__).resolve()))

    neptune.init(**init_cfg)
    params = flatten_dict(cfg["experiment"])
    params["exp_name"] = cfg["main"]["name"]
    neptune.create_experiment(git_info=git_info, params=params, **exp_cfg)


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
    init_neptune(cfg)
    setup_logging(cfg)
    exp_cfg = cfg["experiment"]

    dataset = get_dataset(exp_cfg["dataset"], data_dir=cfg["main"]["data_path"])
    dataset.implicit_(exp_cfg["interaction_pivot"])  # implicit feedback

    model_rng = np.random.default_rng(exp_cfg["model_seed"])
    data_rng = np.random.default_rng(exp_cfg["dataset_seed"])

    train, rest = random_train_test_split(dataset, test_percentage=0.10, rng=data_rng)
    test, valid = random_train_test_split(rest, test_percentage=0.5, rng=data_rng)
    train.max_user_interactions_(exp_cfg["max_user_interactions"], rng=data_rng)

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        log_dataset(name, data)
        _logger.info(f"{name}: {data}")
        _logger.info(f"{name}_hash: {data.hash()}")

    est_class = getattr(estimators, exp_cfg["estimator"])
    est = est_class(**exp_cfg["est_params"], rng=model_rng)
    est.fit(train)
    df = summary(est, train=train, valid=valid, test=test, rng=model_rng)

    log_summary(df.reset_index())
    log_table("summary", df)
    _logger.info(f"Result:\n{df.reset_index()}")


def cmp_pop_lda_mf_v1(template):
    """Compare Popularity, LDA, MF on Movielens 1m"""

    def make_configs(exp, model_params_iter):
        for lr, batch_size, embedding_dim, n_iter in model_params_iter:
            params = exp["est_params"]
            params["learning_rate"] = lr
            params["batch_size"] = batch_size
            params["embedding_dim"] = embedding_dim
            params["n_iter"] = n_iter
            template["experiment"] = exp
            yield template

    estimators = ["LDA4RecEst", "BilinearBPREst", "PopEst"]
    datasets = ["movielens-1m"]
    model_seeds = [3128845410, 2764130162, 4203564202, 2330968889, 3865905591]

    embedding_dims = [4, 8, 12, 16]
    learning_rates = [0.01]
    batch_sizes = [32, 64, 128, 256]
    n_iters_bpr = [10, 25, 50]
    n_iters_lda4rec = [3000, 6000, 10000]

    for estimator, dataset, model_seed in product(estimators, datasets, model_seeds):
        exp = {
            "dataset": dataset,
            "dataset_seed": 1729,  # keep this constant for reproducibility
            "interaction_pivot": 0,
            "model_seed": model_seed,
            "max_user_interactions": 200,
            "estimator": estimator,
            "est_params": {},
        }
        if estimator == "PopEst":
            template["experiment"] = exp
            yield template
        elif estimator == "LDA4RecEst":
            yield from make_configs(
                exp,
                product(learning_rates, batch_sizes, embedding_dims, n_iters_lda4rec),
            )
        elif estimator == "BilinearBPREst":
            yield from make_configs(
                exp,
                product(learning_rates, batch_sizes, embedding_dims, n_iters_bpr),
            )
        else:
            raise RuntimeError(f"Unknown estimator {estimator}!")


@main.command(name="create")
@click.option(
    "-e",
    "--experiment",
    "exp_name",
    required=True,
    type=str,
    help="name of the experiment to create configs for",
)
@click.pass_obj
def create_experiments(cfg: Config, exp_name: str):
    """Create experiment configurations"""
    template = yaml.safe_load(cfg.yaml_content)
    all_experiments = {"cmp_pop_lda_mf_v1": cmp_pop_lda_mf_v1}
    experiments_iter = all_experiments.get(exp_name)
    if experiments_iter is None:
        options = ", ".join(all_experiments.keys())
        msg = f"Unknown experiment: {exp_name}. Choose one of: {options}"
        raise ValueError(msg)

    for idx, experiment in enumerate(experiments_iter(template)):
        with open(cfg.path.parent / Path(f"exp_{idx}.yaml"), "w") as fh:
            yaml.dump(experiment, fh)


if __name__ == "__main__":
    main()
