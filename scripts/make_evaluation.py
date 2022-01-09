#!/usr/bin/env python
"""
Do the evaluation in parallel by running this on every config file.

Run this from root directory:
```
find ./configs -maxdepth 1 -name "exp_*.yaml" -exec pueue add "scripts/make_evaluation.py {}" \;
```
"""
import pickle
from pathlib import Path

import click
import pingouin as pg

import lda4rec.evaluations as lda_eval
from lda4rec.utils import Config

# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

# latent dimensions used in the model
ML_DIM = 64
GB_DIM = 128
AM_DIM = 128  # Check this!!!


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="path to config file",
)
@click.option(
    "-p",
    "--path",
    "eval_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
    help="path to store evaluation files",
)
@click.option("-s", "--silent", "silent", flag_value=True, default=False)
def main(cfg_path, eval_path, silent):
    cfg = Config(Path(cfg_path), silent=silent)
    cfg_exp = cfg["experiment"]
    if not (
        (
            cfg_exp["dataset"] == "movielens-1m"
            and cfg_exp["est_params"]["embedding_dim"] == ML_DIM
        )
        or (
            cfg_exp["dataset"] == "goodbooks"
            and cfg_exp["est_params"]["embedding_dim"] == GB_DIM
        )
        or (
            cfg_exp["dataset"] == "amazon"
            and cfg_exp["est_params"]["embedding_dim"] == AM_DIM
        )
    ):
        return

    train, test, data_rng = lda_eval.get_train_test_data(cfg)
    est = lda_eval.load_model(cfg, train)

    v, t, h, b = est.get_lda_params()

    cfg["result"] = {}
    cfg_res = cfg["result"]

    # first experiment
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
    cfg_res["corr_popularity"] = lda_eval.popularity_ranking_corr(train, b)

    # third experiment
    emp_pops = lda_eval.get_empirical_pops(train)
    cfg_res["corr_conformity_pop"] = lda_eval.conformity_interaction_pop_ranking_corr(
        emp_pops, (1 / t).numpy(), train
    )
    cfg_res["corr_conformity_b"] = lda_eval.conformity_interaction_pop_ranking_corr(
        b, (1 / t).numpy(), train
    )

    # fourth experiment
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
    good_jacs = lda_eval.get_twin_jacs(user_ids, good_twins, test)
    bad_jacs = lda_eval.get_twin_jacs(user_ids, bad_twins, test)
    rnd_jacs = lda_eval.get_twin_jacs(user_ids, rnd_twins, test)
    cfg_res["ttest_user_interaction_good_bad_test"] = pg.ttest(
        good_jacs, bad_jacs, paired=True, alternative="greater"
    )
    cfg_res["ttest_user_interaction_good_rnd_test"] = pg.ttest(
        good_jacs, rnd_jacs, paired=True, alternative="greater"
    )

    file_name = eval_path / Path(f'result_{cfg["main"]["name"]}.pickle')
    with open(file_name, "bw") as fh:
        pickle.dump(cfg, fh)


if __name__ == "__main__":
    main()
