import logging
import os.path
import sys
from collections import UserDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import click
import neptune
import pandas as pd
import yaml
from IPython.core import ultratb
from neptune.utils import get_git_info

from . import __version__, estimators
from .datasets import random_train_test_split
from .evaluations import summary
from .utils import is_cuda_available

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)


_logger = logging.getLogger(__name__)


def relpath_to_abspath(path: Path, anchor_path: Path):
    if path.is_absolute():
        return path
    return (anchor_path / path).resolve()


class Config(UserDict):
    def __init__(self, path: Path, **kwargs):
        super().__init__()

        with open(path, "r") as fh:
            self.yaml_content = fh.read()

        cfg = yaml.safe_load(self.yaml_content)
        timestamp = datetime.now()
        # store name of config file for id later
        cfg["main"].setdefault("name", os.path.splitext(path.name)[0])
        cfg["main"]["log_level"] = getattr(logging, cfg["main"]["log_level"])
        cfg["main"]["timestamp"] = timestamp
        cfg["main"]["timestamp_str"] = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
        cfg["main"].update(kwargs)
        self._resolve_paths(cfg, path.parent)

        sec_cfg = cfg["exp_tracker"]["init"]
        if sec_cfg["api_token"].upper() != "ANONYMOUS":
            # read token file and replace it in config
            with open(Path(sec_cfg["api_token"]).expanduser()) as fh:
                sec_cfg["api_token"] = fh.readline().strip()
        self.data.update(cfg)  # set cfg as own dictionary

    def _resolve_paths(self, cfg: Dict[str, Any], anchor_path: Path):
        for k, v in cfg.items():
            if isinstance(v, dict):
                self._resolve_paths(v, anchor_path)
            elif k.endswith("_path"):
                cfg[k] = relpath_to_abspath(Path(v).expanduser(), anchor_path)


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
    cfg = Config(Path(cfg_path), silent=silent)

    init_cfg = cfg["init"]
    exp_cfg = cfg["create_experiment"]
    # needs to be determined explicitly because of `console_scripts`
    git_info = get_git_info(str(Path(__file__).resolve()))

    neptune.init(**init_cfg)
    return neptune.create_experiment(
        git_info=git_info, params=flatten_dict(params), **exp_cfg
    )

    logging.basicConfig(
        stream=sys.stdout,
        level=cfg["main"]["log_level"],
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # set up additional file logger
    log_file = cfg["main"]["log_path"] / Path(
        f"{cfg['main']['name']}_{cfg['main']['timestamp_str']}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(cfg["main"]["log_level"])
    # add handler to the root logger of this package
    logging.getLogger(__name__.split(".")[0]).addHandler(fh)

    _logger.info(f"Configuration:\n{cfg.yaml_content}")
    ctx.obj = cfg  # pass to others


def synth_data_name(generator: Generator) -> Path:
    return Path(f"sd-{generator.uid()}.pickle")


@main.command(name="generate")
@click.option("-f", "--force", "force", flag_value=True, default=False)
@click.pass_obj
def generate_data(cfg: Config, force: bool):
    generator = Generator(**cfg["generator"])
    dest_file = cfg["main"]["data_path"] / synth_data_name(generator)
    if dest_file.is_file() and not force:
        _logger.info(f"File {dest_file} already exists.")
    else:
        synth_data = generator.make_data(silent=cfg["main"]["silent"])
        synth_data.dump(dest_file)


@main.command(name="evaluate")
@click.option("-f", "--force", "force", flag_value=True, default=False)
@click.pass_context
def evaluate_benchmarks(ctx, force: bool):
    ctx.invoke(generate_data, force=force)
    cfg = ctx.obj

    generator = Generator(**cfg["generator"])
    synth_data_path = cfg["main"]["data_path"] / synth_data_name(generator)
    synth_data = SynthData.load(synth_data_path)

    dfs = []
    for threshold in (0, 4):
        _logger.info(f"Using threshold {threshold} for data")
        data = synth_data.interactions.implicit(threshold)
        train, test = random_train_test_split(data)

        for est_str, est_cfg in cfg["estimators"].items_batch():
            _logger.info(f"Benchmarking {est_str}...")

            with exp_tracker.init(cfg["exp_tracker"], params=cfg):
                exp_tracker.set_property("threshold", threshold)
                exp_tracker.set_property("model", est_str)
                exp_tracker.set_property("data", synth_data_name(generator))
                exp_tracker.set_property("generator", generator.uid())

                model_cls = getattr(models, est_cfg["model"])
                model = model_cls(
                    data._n_users, data._n_items, **est_cfg["model_params"]
                )

                est_cls = getattr(estimators, est_cfg["estimator"])
                est = est_cls(
                    model=model,
                    use_cuda=is_cuda_available(),
                    callback=lambda x: exp_tracker.log_metric("loss", x),
                    **est_cfg["estimator_params"],
                )
                est.fit(train)
                df = summary(est, train=train, test=test).reset_index()
                df.insert(0, "model", est_str)
                df.insert(1, "threshold", threshold)
                dfs.append(df)

                exp_tracker.log_table("summary", df)
                exp_tracker.log_summary(df)
                remote_path = Path("synth_data") / synth_data_path.name
                exp_tracker.log_artifact(str(synth_data_path), str(remote_path))

    df = pd.concat(dfs)
    df.insert(0, "generator", generator.uid())
    df.insert(0, "data", synth_data_name(generator))
    df.insert(0, "config", cfg["main"]["name"])
    df.insert(0, "version", __version__)
    df.insert(0, "timestamp", cfg["main"]["timestamp"])
    result_file = Path(f"df-{cfg['main']['name']}_{cfg['main']['timestamp_str']}.csv")
    result_dest = cfg["main"]["result_path"] / result_file
    _logger.info(f"Storing results under {result_dest}")
    _logger.info(df)
    df.to_csv(result_dest, index=False)


if __name__ == "__main__":
    main()
