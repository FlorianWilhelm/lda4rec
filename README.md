# LDA4Rec / LDAext

![LDA4Rec](docs/gfx/lda4rec_601x132.png?raw=true)

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

Accompanying source code to the paper "Matrix Factorization for Collaborative Filtering is just Solving an
Adjoint Latent Dirichlet Allocation Model After All" by Florian Wilhelm and "An Interpretable Model for Collaborative Filtering Using
an Extended Latent Dirichlet Allocation Approach" by Florian Wilhelm, Marisa Mohr and Lien Michiels. Check out
git tag v1.0 for the former and v2.0 for the latter.

The preprint of "Matrix Factorization for Collaborative Filtering is just Solving an Adjoint Latent Dirichlet Allocation Model After All"
can be found [here](docs/lda4rec_fwilhelm_prepint.pdf) along with the following statement:

> "© Florian Wilhelm 2021. This is the author's version of the work. It is posted here for
your personal use. Not for redistribution. The definitive version was published
in RecSys '21: Fifteenth ACM Conference on Recommender Systems Proceedings, https://doi.org/10.1145/3460231.3474266."

The preprint of "An Interpretable Model for Collaborative Filtering Using an Extended Latent Dirichlet Allocation Approach"
can be found [here](docs/ldaext_fwilhelm_preprint.pdf) and the final paper [here](https://journals.flvc.org/FLAIRS/article/view/130567).

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
lda4rec -c configs/default.yaml create
```
Check out `cli.py` for more details.


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

## How to Cite

Please cite LDA4Rec/LDAext if it helps your research. You can use the following BibTeX entry:

```
@inproceedings{wilhelm2021lda4rec,
author = {Wilhelm, Florian},
title = {Matrix Factorization for Collaborative Filtering Is Just Solving an Adjoint Latent Dirichlet Allocation Model After All},
year = {2021},
month = sep,
isbn = {978-1-4503-8458-2/21/09},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3460231.3474266},
doi = {10.1145/3460231.3474266},
booktitle = {Fifteenth ACM Conference on Recommender Systems},
numpages = {8},
location = {Amsterdam, Netherlands},
series = {RecSys '21}
}
@article{Wilhelm_Mohr_Michiels_2022, 
title={An Interpretable Model for Collaborative Filtering Using an Extended Latent Dirichlet Allocation Approach}, 
volume={35}, 
url={https://journals.flvc.org/FLAIRS/article/view/130567}, 
DOI={10.32473/flairs.v35i.130567}, 
abstractNote={With the increasing use of AI and ML-based systems, interpretability is becoming an increasingly important issue to ensure user trust and safety. This also applies to the area of recommender systems, where methods based on matrix factorization (MF) are among the most popular methods for collaborative filtering tasks with implicit feedback. Despite their simplicity, the latent factors of users and items lack interpretability in the case of the effective, unconstrained MF-based methods. In this work, we propose an extended latent Dirichlet Allocation model (LDAext) that has interpretable parameters such as user cohorts of item preferences and the affiliation of a user with different cohorts. We prove a theorem on how to transform the factors of an unconstrained MF model into the parameters of LDAext. Using this theoretical connection, we train an MF model on different real-world data sets, transform the latent factors into the parameters of LDAext and test their interpretation in several experiments for plausibility. Our experiments confirm the interpretability of the transformed parameters and thus demonstrate the usefulness of our proposed approach.}, 
journal={The International FLAIRS Conference Proceedings}, 
author={Wilhelm, Florian and Mohr, Marisa and Michiels, Lien}, 
year={2022}, 
month={May} 
}
```

## License

This sourcecode is [AGPL-3-only](LICENSE.txt) licensed. If you require a more permissive licence, e.g. for
commercial reasons, contact me to obtain a licence for your business.

<!-- pyscaffold-notes -->

## Acknowledgement

Special thanks goes to [Du Phan](https://github.com/fehiepsi) and [Fritz Obermeyer](https://github.com/fritzo) from the [(Num)Pyro](https://github.com/pyro-ppl) project for their kind help and helpful comments on my code.

## Note

This project has been set up using [PyScaffold] 4.0 and the [dsproject extension] 0.6.
Some source code was taken from [Spotlight] (MIT-licensed) by Maciej Kula as well as [lrann] (MIT-licensed) by
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
