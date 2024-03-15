# Dynamic dispatch waves problem
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/leonlan/dynamic-dispatch-waves/actions/workflows/CI.yml/badge.svg)](https://github.com/leonlan/dynamic-dispatch-waves/actions/workflows/CI.yml)

This repository hosts all code used in the paper:

> Lan, L., van Doorn, J., Wouda, N. A., Rijal, A., & Bhulai, S. (2024). An iterative sample scenario approach for the dynamic dispatch waves problem. Transportation Science, forthcoming. https://doi.org/10.1287/trsc.2023.0111.

In the dynamic dispatch waves problem (DDWP), a set of delivery requests arrive at each epoch, which must be served before the end of the planning horizon. 
At each decision epoch, it must be decided which requests to dispatch in the current epoch (and how to route them), and which requests to postpone to consolidate with future requests that arrive in later epochs.

Our work proposes _iterative conditional dispatch_ (ICD), an iterative solution construction procedure based on a sample scenario approach. ICD iteratively solves sample scenarios to classify requests to be dispatched, postponed, or undecided. See [our paper](#paper) for more information about the DDWP and the implemented algorithms.


## Installation

To install this repository, you need the following:
- Python version 3.9 or higher;
- [Poetry](https://python-poetry.org/) version 1.2 or higher; and
- A modern C++ compiler that supports C++20.

Then, run the following command:

```bash
poetry install
```

This will install all dependencies, including a custom version of PyVRP extended with support for dispatch windows. See the [`pyvrp`](https://github.com/leonlan/dynamic-dispatch-waves/tree/pyvrp) branch for more details, as well as Appendix B in the paper. 

## Usage

This repository includes an environment that models the DDWP.
There are two specialized constructors for the environment: one variant that was used during the [EURO Meets NeurIPS 2022 Vehicle Routing Competition](https://euro-neurips-vrp-2022.challenges.ortec.com/), and another variant that was used in our paper. 
The environment requires a sampling method (see `sampling/`), which describes how future, unknown requests are sampled.
Moreover, a number of solution methods can be found under `agents/`.

### Example: solve an instance from the paper

To solve an instance of the DDWP from our paper, you can run the following example command:

``` bash
poetry run benchmark instances/hg/C1_10_1.txt \
--env_seed 1 \
--num_requests_per_epoch 75 75 75 75 75 75 75 75 \
--sampling_method TW2 \
--agent_config_loc configs/icd-hamming-distance.toml \
--epoch_tlim 5 \
--strategy_tlim 3 
```

This solves the an instance of the DDWP problem using the static VRP instance `C_10_1`.
In each epoch, an expected number of 75 requests are sampled with two-hour time windows.
The ICD with hamming distance consensus function is used to solve this instance.
Each epoch lasts five seconds. During this, three seconds are used by ICD to select which requests to dispatch, and the remaining two seconds are used for final route planning.

### Example: solve an instance from EURO Meets NeurIPS

Here's an example command to solve an instance from the EURO Meets NeurIPS competition:

``` bash
poetry run euro_neurips instances/ortec/ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35.txt \
--env_seed 0 \
--agent_config_loc configs/icd-double-threshold.toml \
--strategy_tlim 3 \
--epoch_tlim 5 
```

This solves an DDWP instance that is created using the static VRP instance `ORTEC-VRPTW-ASYM-0bdff870-d1-n458-k35` and environment seed 0.
The ICD double threshold configuration is used to solve this instance.
Each epoch lasts five seconds. During this, three seconds are used by ICD to select which requests to dispatch, and the remaining two seconds are used for final route planning.

## Paper

For more details see our paper *[An iterative sample scenario approach for the dynamic dispatch waves problem](https://pubsonline.informs.org/doi/10.1287/trsc.2023.0111)*. If this code is useful to you, please consider citing our work:

``` bibtex
@article{Lan_et_al_2024,
  title = {An iterative sample scenario approach for the dynamic dispatch waves problem},
  author = {Lan, Leon and {van Doorn}, Jasper and Wouda, Niels A. and Rijal, Arpan and Bhulai, Sandjai},
  doi = {10.1287/trsc.2023.0111},
  year = {2024},
  publisher = {INFORMS},
  journal = {Transportation Science},
}
```

A preprint version of our paper is available on [arXiv](https://arxiv.org/abs/2308.14476).
