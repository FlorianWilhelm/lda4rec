# LDA4Rec

Matrix Factorization for Collaborative Filtering is actually an Approximation to a Latent Dirichlet Allocation Problem

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

## Running experiments

First check out and adapt the default config `configs/default.yaml` which also serves as a template.
The create the configs of the experiments with:
```
lda4rec -c configs/default.yaml create
```
which populates the `configs` folders with tons of experiments yaml files.
Execute a single experiment with:
```
lda4rec -c configs/exp_0.yaml run
```
or run all experiments with [pueue] for full control over parallelism:
```
find ./configs -name "*.yaml" -maxdepth 1 -exec pueue add "lda4rec -c {} run" \;
```

## Cloud Setup

Commands for setting up an Ubuntu 20.10 VM with at least 20 GiB of HD:
```
sudo apt-get install -y build-essential
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
cargo install pueue
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O
sh Miniconda3-latest-Linux-x86_64.sh
source /home/fwilhelm/.bashrc
git clone https://github.com/FlorianWilhelm/lda4rec.git # and TOKEN as password !!!
cd lda4rec
conda env create -f environment.yml
conda activate lda4rec
pueued -d
vim ~/.neptune_api_token # and copy it over
```
Then create and run all experiments as defined above:
```
export OMP_NUM_THREADS=1
lda4rec -c configs/default.yaml create
find ./configs -name "*.yaml" -maxdepth 1 -exec pueue add "lda4rec -c {} run" \;
```


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
[Spotlight]: https://github.com/maciejkula/spotlight
[pueue]: https://github.com/Nukesor/pueue
[lrann]: https://github.com/FlorianWilhelm/lrann
[neptune.ai]: https://neptune.ai/
