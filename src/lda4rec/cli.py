import logging
import sys
from pathlib import Path

import click
import neptune
import numpy as np
from IPython.core import ultratb
from neptune.utils import get_git_info
from neptunecontrib.api.table import log_table

from . import __version__, estimators
from .datasets import get_dataset, random_train_test_split
from .evaluations import summary
from .utils import Config, NeptuneLogHandler, flatten_dict, log_dataset, log_summary

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)


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

    # This handles only configuration, Neptune initialization and logging
    cfg = Config(Path(cfg_path), silent=silent)
    neptune_cfg = cfg["neptune"]
    init_cfg = neptune_cfg["init"]
    exp_cfg = neptune_cfg["create_experiment"]
    # needs to be determined explicitly because of `console_scripts`
    git_info = get_git_info(str(Path(__file__).resolve()))

    neptune.init(**init_cfg)
    neptune.create_experiment(
        git_info=git_info, params=flatten_dict(cfg["experiment"]), **exp_cfg
    )

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
    # set up Neptune text logger
    nh = NeptuneLogHandler(cfg["main"]["name"])
    nh.setLevel(cfg["main"]["log_level"])
    logging.getLogger(pkg_logger).addHandler(nh)

    _logger.info(f"Configuration:\n{cfg.yaml_content}")
    ctx.obj = cfg  # pass Config to other commands


@main.command(name="run")
@click.pass_obj
def run_experiment(cfg: Config):
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


if __name__ == "__main__":
    main()
