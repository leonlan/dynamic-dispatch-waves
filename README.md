
# Dynamic dispatch waves problem
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/leonlan/dynamic-dispatch-waves/actions/workflows/CI.yml/badge.svg)](https://github.com/leonlan/dynamic-dispatch-waves/actions/workflows/CI.yml)

This repository hosts all code used to solve the *dynamic dispatch waves problem* (DDWP). 

In the DDWP, a set of delivery requests arrive at each epoch, which must be served before the end of the planning horizon. 
At each decision epoch, it must be decided which requests to dispatch in the current epoch (and how to route them), and which requests to postpone to consolidate with future requests that arrive in later epochs.

See [our paper](#paper) for more information about the DDWP and the implemented algorithms.


## Installation
Make sure to have [Poetry](https://python-poetry.org/) installed with version 1.2 or higher. 
The following command will then install all necessary dependencies:

```bash
poetry install
```

If you don't have Poetry installed, make sure that you have Python 3.9 or higher and install the packages indicated in the `pyproject.toml` file. 

## Usage

This repository includes an environment that models the DDWP.
There are two specialized constructors for the environment: one variant that was used during the [EURO Meets NeurIPS 2022 Vehicle Routing Competition](https://euro-neurips-vrp-2022.challenges.ortec.com/), and another variant that was used in our paper. 
The environment requires a sampling method (see `sampling/`), which describes how future, unknown requests are sampled.
Moreover, a number of solution methods can be found under `agents/`.

To solve an instance of the DDWP, you can use the `benchmark.py` script. Here's an example:

``` bash
poetry run benchmark instances/ortec/ORTEC-VRPTW-ASYM-01829532-d1-n324-k22.txt \
    --environment euro_neurips --sampling_method euro_neurips --env_seed 1 \
    --agent_config_loc configs/icd-double-threshold.toml --agent_seed 2 \
    --epoch_tlim 5
```

This solves the an instance of the DDWP problem using the static VRP instance `ORTEC-VRPTW-ASYM-01829532-d1-n324-k22` for sampling future requests.
It follows the EURO-NeurIPS environment and sampling procedure with seed 1. 
It uses the iterative conditional dispatch algorithm with double threshold consensus function and seed 2.
Each epoch has a time limit of five seconds, which is the maximum time that an algorithm spend in a single epoch before it must return a solution to the environment.


## Experiments

TODO

## Paper

For more details about the DDWP, see our paper *[An iterative conditional dispatch algorithm for the dynamic dispatch waves problem](https://doi.org/10.48550/arXiv.2308.14476)*. If this code is useful for your work, please consider citing our work:

``` bibtex
@misc{Lan2023,
  title = {An iterative conditional dispatch algorithm for the dynamic dispatch waves problem}, 
  author = {Leon Lan and Jasper van Doorn and Niels A. Wouda and Arpan Rijal and Sandjai Bhulai},
  year = {2023},
  eprint = {arXiv:2308.14476},
  url = {https://doi.org/10.48550/arXiv.2308.14476}
}
```
