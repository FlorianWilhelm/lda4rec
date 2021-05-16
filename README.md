# LDA4Rec Next Generation

Matrix Factorization for collaborative filtering is just solving an adjoint Latent Dirichlet Allocation model after all.

## Ideas & Next Steps

* Examine the optimality between the opimization problem MF-BPR and LDA4Rec from a theoretical perspective.
* Evaluate the hyperparameters more for LDA4Rec. So far no extensive evaluation was performed.
  Maybe also replace LogNormal with Gamma.
* Extend LDA4Rec by introducing a [Dirichlet Process](https://pyro.ai/examples/dirichlet_process_mixture.html) allowing
  to only define maximum latent dimension, thus making the model easier to apply.
* Evaluate the non-sampling approach for prediction further that is as fast as MF-BPR.
  Initial implementation exists in `estimators.py`


## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `lda4rec` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate lda4rec
   ```
3. (optionally) get a free [neptune.ai] account for experiment tracking and save the api token
   under `~/.neptune_api_token` (default).

## Running Experiments

First check out and adapt the default experiment config `configs/default.yaml` and run it with:
```
lda4rec -c configs/default.yaml run
```
A config like `configs/default.yaml` can also be used as a template to create an experiment set with:
```
lda4rec -c configs/default.yaml create -ds movielens-100k
```
using the Movielens-100k dataset. Check out `cli.py` for more details.


## Cloud Setup

Commands for setting up an Ubuntu 20.10 VM with at least 20 GiB of HD on e.g. a GCP c2-standard-30 instance:
```
tmux
sudo apt-get install -y build-essential
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
cargo install pueue
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
git clone https://github.com/FlorianWilhelm/lda4rec.git
cd lda4rec
conda env create -f environment.yml
conda activate lda4rec
vim ~/.neptune_api_token # and copy it over
```
Then create and run all experiments for full control over parallelism with [pueue]:
```
pueued -d # only once to start the daemon
pueue parallel 10
export OMP_NUM_THREADS=4  # to limit then number of threads per model
lda4rec -c configs/default.yaml create # to create the config files
find ./configs -maxdepth 1 -name "exp_*.yaml" -exec pueue add "lda4rec -c {} run" \; -exec sleep 30 \;
```
Remark: `-exec sleep 30` avoids race condition when reading datasets if parallelism is too high.


## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n lda4rec -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data                    <- Downloaded datasets will be stored here.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── logs                    <- Generated logs are collected here.
├── results                 <- Results as exported from neptune.ai.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
│                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── lda4rec             <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## How to Cite

Please cite LDA4Rec if it helps your research. You can use the following BibTeX entry:

```
@inproceedings{wilhelm2021lda4rec,
author = {Wilhelm, Florian},
title = {Matrix Factorization for Collaborative Filtering Is Just Solving an Adjoint Latent Dirichlet Allocation Model After All},
year = {2021},
% isbn = {9781450375832},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
% url = {https://doi.org/10.1145/3383313.3412240},
% doi = {10.1145/3383313.3412240},
booktitle = {Fifteenth ACM Conference on Recommender Systems},
% pages = {13–22},
% numpages = {10},
keywords = {Conversational Recommendation, Critiquing},
% location = {Virtual Event, Brazil},
series = {RecSys '21}
}
```

The preprint can be found [here](docs/lda4rec_fwilhelm_prepint.pdf).

## Note

This project has been set up using [PyScaffold] 4.0 and the [dsproject extension] 0.6.
Some code was taken from [Spotlight] (MIT-licensed) by Maciej Kula as well as [lrann] (MIT-Licensed) by
Florian Wilhelm and Marcel Kurovski.

[PyScaffold]: https://pyscaffold.org/
[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
[pueue]: https://github.com/Nukesor/pueue
[neptune.ai]: https://neptune.ai/
[Spotlight]: https://github.com/maciejkula/spotlight
[lrann]: https://github.com/FlorianWilhelm/lrann
