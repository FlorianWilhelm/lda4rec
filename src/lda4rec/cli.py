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
import pickle
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import click
import neptune.new as neptune
import numpy as np
import pingouin as pg
import yaml
from neptune.new.types import File
from neptune.utils import get_git_info

from . import __version__, estimators
from . import evaluations as lda_eval
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
    data, train, test, data_rng = lda_eval.get_train_test_data(cfg)

    for name, data in [("train", train), ("test", test)]:
        log_dataset(name, data)
        _logger.info(f"{name}: {data}")
        _logger.info(f"{name}_hash: {data.hash()}")

    est_class = getattr(estimators, exp_cfg["estimator"])
    try:
        est = est_class(**exp_cfg["est_params"], rng=model_rng)
        _logger.info(f"Fitting estimator {exp_cfg['estimator']}...")
        est.fit(train)
        _logger.info("Evaluating...")
        df = lda_eval.summary(est, train=train, test=test, eval_train=False)

        log_summary(df.reset_index())
        run["summary/df"].upload(File.as_html(df))
        _logger.info(f"Result:\n{df.reset_index()}")
    except Exception:
        run["sys/tags"].add("failure")
        raise
    else:
        run["sys/tags"].add("success")

    mcfg = cfg["main"]
    model_path = Path(mcfg["model_path"]) / Path(
        f"{mcfg['name']}_{exp_cfg['estimator']}_{mcfg['timestamp_str']}.model"
    )
    _logger.info(f"Storing model as {model_path}...")
    est.save(model_path)
    _logger.info("Experiment finished successfully!")


def experiments_gen(template):
    """Generate different experiment config setups"""

    def make_configs(template, param_names, model_params_iter):
        for params in model_params_iter:
            exp_temp = deepcopy(template)
            for name, param in zip(param_names, params):
                exp_temp["experiment"]["est_params"][name] = param
            yield exp_temp

    estimators = [
        "MFEst",
        # "LDA4RecEst",
        # "HierLDA4RecEst",
        # "HierVarLDA4RecEst",
    ]
    datasets = ["amazon", "movielens-1m", "goodbooks"]
    model_seeds = [1729, 1981, 666, 234829, 92349402]

    learning_rates = [0.001]
    batch_sizes = [256]
    train_test_splits = ["items_per_user_train_test_split"]
    # ^ "random_train_test_split"

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
        if estimator == "MFEst":
            if dataset == "movielens-1m":
                embedding_dims = [64]
            elif dataset == "amazon":
                embedding_dims = [256]
            elif dataset == "goodbooks":
                embedding_dims = [256]
            else:
                raise RuntimeError(f"Unknown dataset {dataset}!")

            params = {
                "embedding_dim": embedding_dims,
                "learning_rate": learning_rates,
                "batch_size": batch_sizes,
                "n_iter": [200],
            }
            yield from make_configs(exp_temp, params.keys(), product(*params.values()))
        else:
            raise RuntimeError(f"Unknown estimator {estimator}!")


@main.command(name="eval")
@click.pass_obj
def evaluate(cfg):
    setup_logging(cfg)
    _logger.info("Starting the evaluation...")
    data, train, test, data_rng = lda_eval.get_train_test_data(cfg)
    est = lda_eval.load_model(cfg, train)

    v, t, h, b = est.get_lda_params()

    cfg["result"] = {}
    cfg_res = cfg["result"]

    # first experiment
    _logger.info("Evaluating experiment 1...")
    user_ids, log_probs = lda_eval.cohort_user_interaction_log_probs(
        train, v, h, rng=data_rng
    )
    cfg_res["ttest_cohort_user_interaction_train"] = pg.ttest(
        log_probs[:, 1], log_probs[:, 0], paired=True, alternative="greater"
    )
    user_ids, log_probs = lda_eval.cohort_user_interaction_log_probs(
        test, v, h, rng=data_rng
    )
    cfg_res["ttest_cohort_user_interaction_test"] = pg.ttest(
        log_probs[:, 1], log_probs[:, 0], paired=True, alternative="greater"
    )

    # second experiment
    _logger.info("Evaluating experiment 2...")
    cfg_res["corr_popularity_train"] = lda_eval.popularity_ranking_corr(train, b)
    cfg_res["corr_popularity_data"] = lda_eval.popularity_ranking_corr(data, b)

    # third experiment
    _logger.info("Evaluating experiment 3...")
    emp_pops = lda_eval.get_empirical_pops(train)
    cfg_res[
        "corr_conformity_pop_train"
    ] = lda_eval.conformity_interaction_pop_ranking_corr(
        emp_pops, (1 / t).numpy(), train
    )
    emp_pops = lda_eval.get_empirical_pops(data)
    cfg_res[
        "corr_conformity_pop_data"
    ] = lda_eval.conformity_interaction_pop_ranking_corr(
        emp_pops, (1 / t).numpy(), train
    )
    emp_pops = lda_eval.get_empirical_pops(data)
    cfg_res[
        "corr_conformity_pop_data_all"
    ] = lda_eval.conformity_interaction_pop_ranking_corr(
        emp_pops, (1 / t).numpy(), data
    )
    cfg_res["corr_conformity_b"] = lda_eval.conformity_interaction_pop_ranking_corr(
        b, (1 / t).numpy(), train
    )

    # fourth experiment
    _logger.info("Evaluating experiment 4...")
    user_ids, good_twins, bad_twins, rnd_twins = lda_eval.find_good_bad_rnd_twins(
        v, n_users=2000, rng=data_rng
    )
    good_jacs = lda_eval.get_twin_jacs(user_ids, good_twins, train)
    bad_jacs = lda_eval.get_twin_jacs(user_ids, bad_twins, train)
    rnd_jacs = lda_eval.get_twin_jacs(user_ids, rnd_twins, train)
    cfg_res["ttest_user_interaction_good_bad_train"] = pg.ttest(
        good_jacs, bad_jacs, paired=True, alternative="greater"
    )
    cfg_res["ttest_user_interaction_good_rnd_train"] = pg.ttest(
        good_jacs, rnd_jacs, paired=True, alternative="greater"
    )
    cfg_res["ttest_user_interaction_rnd_bad_train"] = pg.ttest(
        rnd_jacs, bad_jacs, paired=True, alternative="greater"
    )
    good_jacs = lda_eval.get_twin_jacs(user_ids, good_twins, test)
    bad_jacs = lda_eval.get_twin_jacs(user_ids, bad_twins, test)
    rnd_jacs = lda_eval.get_twin_jacs(user_ids, rnd_twins, test)
    cfg_res["ttest_user_interaction_good_bad_test"] = pg.ttest(
        good_jacs, bad_jacs, paired=True, alternative="greater"
    )
    cfg_res["ttest_user_interaction_good_rnd_test"] = pg.ttest(
        good_jacs, rnd_jacs, paired=True, alternative="greater"
    )
    cfg_res["ttest_user_interaction_rnd_bad_test"] = pg.ttest(
        rnd_jacs, bad_jacs, paired=True, alternative="greater"
    )

    eval_path = Path(cfg["main"]["eval_path"])
    file_name = eval_path / Path(f'result_{cfg["main"]["name"]}.pickle')
    _logger.info("Writing results...")
    with open(file_name, "bw") as fh:
        pickle.dump(cfg, fh)
    _logger.info("Evaluation ended successfully!")


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
