# Dynamic dispatch waves problem
This repository hosts all code used to solve the *dynamic dispatch waves problem* (DDWP). 

In the DDWP, a set of unknown delivery requests arrive at each epoch, which must be served before the end of the planning horizon. 
At each decision epoch, it must be decided which requests to dispatch in the current epoch (and how to route them), and which requests to postpone to consolidate with future requests that arrive in later epochs.

See [our paper](#paper) for more information about the DDWP. 


## Installation
Make sure to have [Poetry](https://python-poetry.org/) installed with version 1.2 or higher. 
The following command will then install all necessary dependencies:

```bash
poetry install
```


If you don't have Poetry installed, make sure that you have Python 3.9 or higher and install the packages indicated in the `pyproject.toml` file. 

## Usage

This repository includes several different environments (i.e., models) of the DDWP, each of which implement the `Environment` protocol. 
All environments can be found under `environments/`.
Morover, agents (i.e., algorithms) to solve the DDWP implement the `Agent` protocol, which can be found under `agents/`.

To solve an instance of the dynamic dispatch waves problem, you can use the script `benchmark.py`. Here's an example:

``` bash
poetry run benchmark ORTEC-VRPTW-ASYM-01829532-d1-n324-k22.txt \
    --environment euro_neurips --env_seed 1 --agent greedy --agent_seed 2 \
    --epoch_tlim 5
```

This solves the instance `ORTEC-VRPTW-ASYM-01829532-d1-n324-k22` within the `EuroNeurips` environment with seed 1. 
It uses a greedy agent with seed 2 as solver, which simply dispatches all requests at each decision epoch.
Each epoch has a time limit of five seconds, which is the maximum time that an agent can use before it must return a solution to the environment.

## Paper

For more details about the DDWP, please see our paper "An iterative conditional dispatch algorithm for the dynamic dispatch waves problem" (TODO). If this code is useful for your work, please consider citing our work:

``` bibtex
@article{Lan2023,
  title = {An iterative conditional dispatch algorithm for the dynamic dispatch waves problem},
  author = {Lan, Leon and {van Doorn}, Jasper and Wouda, Niels A. and Rijal, Arpan and Bhulai, Sandjai},
  year = {2023}
}
```
