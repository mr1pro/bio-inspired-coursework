# Training a neural network with particle swarm optimisation

F21BC Biologically Inspired Computation coursework. A feedforward neural
network and particle swarm optimisation (PSO), both written from scratch in
NumPy, are coupled so that PSO finds the network's weights and biases. The
network is trained to tell genuine from forged banknotes
([UCI banknote authentication data](https://archive.ics.uci.edu/dataset/267/banknote+authentication)).

No neural-network or PSO library is used. NumPy does the arithmetic; pandas,
SciPy and matplotlib are used only to analyse and plot results.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

python main.py            # train one network (about 0.1 s) and print its test results
python -m pytest -q       # run the tests
```

Everything is run from this folder. Python 3.9 or newer.

`python main.py` prints something like

```
Data: 1372 samples -> 960 training, 412 test (class 1: 0.445 of training, 0.444 of test)
Network: 4-8(tanh)-1(logistic)   parameters: 49   particle length: 49
PSO: 1000 fitness evaluations, best fitness 1.0000
Training set: accuracy 1.0000  precision 1.0000  recall 1.0000  F1 1.0000   (TP 427, TN 533, FP 0, FN 0)
Test set    : accuracy 1.0000  precision 1.0000  recall 1.0000  F1 1.0000   (TP 183, TN 229, FP 0, FN 0)
```

## Changing the network and the swarm

Every hyperparameter is a command-line option (`python main.py --help` lists them all):

```bash
python main.py --hidden 16 --activations relu                  # one hidden layer of 16 ReLU units
python main.py --hidden 8 8 --activations tanh logistic        # two hidden layers, one activation each
python main.py --hidden                                        # no hidden layer
python main.py --swarm-size 50 --iterations 20                 # same 1000 evaluations, spent differently
python main.py --alpha 0.72 --beta 1.5 --gamma 1.5 --delta 0   # coefficients of Algorithm 39
python main.py --informants 5 --informant-strategy nearest     # random | nearest | ring
python main.py --boundary clamp --bound 2                      # none | clamp | reflect | wrap | random
python main.py --search-activations                            # PSO also chooses the hidden activations
python main.py --random-network                                # evaluate an untrained network
python main.py --seed 7 --plot                                 # another run; saves results/convergence.png
```

## Reproducing the experiments in the report

```bash
python run_experiments.py --list                      # what is available
python run_experiments.py --experiment all --runs 30  # everything, about 6 minutes on 4 cores
python run_experiments.py --experiment budget         # or one experiment at a time
python analyse_results.py                             # tables, significance tests, figures
```

Runs are seeded (seeds 0 to 29 for each setting), so the numbers in
`results/` are reproduced exactly. Output:

| Path | Content |
|---|---|
| `results/raw/` | one CSV row per run, the configuration behind each label, and per-iteration curves |
| `results/tables/` | mean, standard deviation, Mann-Whitney U p-values (Holm-corrected) and effect sizes |
| `results/summary.md` | all tables in one readable file |
| `results/figures/` | the figures used in the report |

## How the code is organised

```
main.py                 train and test one network from the command line
run_experiments.py      run the experiment suites (30 seeded runs per setting, in parallel)
analyse_results.py      tables, statistical tests and figures

src/mlp/                THE NEURAL NETWORK
  activations.py          logistic, tanh, ReLU (+ leaky ReLU, ELU, linear, softmax)
  layer.py                one layer: weights, biases, activation, forward()
  network.py              list of layers; forward(); get/set all parameters as one vector
  builder.py              MLPBuilder: builds a network from {"hidden_layers", "activations", "task"}
  losses.py               cross-entropy and MSE (alternative fitness measures)

src/pso/                PARTICLE SWARM OPTIMISATION (knows nothing about neural networks)
  population.py           the swarm and the main loop: Algorithm 39, tagged line by line
  particle.py             position, velocity, personal best, informants; velocity and position updates
  informant_strategy.py   who informs whom: random, nearest neighbours, ring
  boundary.py             boundary handling: none, clamp, reflect, wrap, random
  fitness_function.py     the one-method interface PSO optimises

src/adapter/            THE COUPLING
  mlp_fitness_adapter.py        particle -> network weights -> training accuracy -> fitness
  activation_search_adapter.py  the same, with extra genes that select the activations

src/data/dataset.py     loading, stratified split, stratified k-fold, standardisation
src/evaluation/metrics.py   accuracy, precision, recall, F1, confusion counts

experiments/            baseline configuration and the definition of every experiment
tests/                  70 unit tests (pytest)
data/                   the UCI data file
```

### Where Algorithm 39 is in the code

PSO follows Algorithm 39 of Luke, *Essentials of Metaheuristics* (2nd ed.,
Section 3.5). Comments tagged `[A39 Lnn]` give the line of the algorithm that
each piece of code implements; the algorithm itself is reproduced at the top
of `src/pso/population.py`.

| Algorithm 39 | Code |
|---|---|
| lines 1-6, parameters | arguments of `Population.__init__` |
| lines 7-10, create the swarm | `Population.initialise` |
| lines 12-13, assess fitness | `Population.evaluate_particles`, `Particle.evaluate` |
| lines 14-15, track the best | `Population.update_global_best` |
| line 17, personal best | `Particle.update_personal_best` |
| line 18, informants' best | `Population.assign_informants`, `Particle.get_informants_best` |
| lines 20-24, new velocity | `Particle.update_velocity` |
| lines 25-26, move | `Particle.update_position` |
| lines 11, 27, 28, loop and result | `Population.optimise` |

### How PSO and the network are coupled

A particle's position is a vector holding every weight and bias of the
network, layer by layer (weights then biases). To score a particle,
`MLPFitnessAdapter.evaluate` writes the vector into the network with
`Network.set_parameters`, classifies the training set, and returns the
training accuracy. PSO maximises that number. The test set is only used after
PSO has finished.

## Data

`data/data_banknote_authentication.txt` is the file distributed by the UCI
Machine Learning Repository (Lohweg, 2012, https://doi.org/10.24432/C55P57,
CC BY 4.0): 1372 rows, four features and a 0/1 class label. It is split 70/30
with the class proportions kept, and the inputs are standardised using the
training set's mean and standard deviation.
