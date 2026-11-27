"""Assemble the text, tables and figure list of the report.

Every number in the report is computed here from the files in results/ (the
per-run tables in results/raw and the test results in results/tables), so the
report cannot drift away from the experiments. The output,
report/content.json, is turned into a Word document by build_docx.js.

Inline markup understood by build_docx.js:
    **bold**   *italic*   `code`   ^{superscript}   _{subscript}
    {ast} stands for a literal asterisk
"""

import json
import os
import re
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, "results", "tables")
RAW = os.path.join(ROOT, "results", "raw")
FIGURES = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, ROOT)

RUNS = {name[:-4]: pd.read_csv(os.path.join(RAW, name))
        for name in os.listdir(RAW) if name.endswith(".csv")}
T = {name[:-4]: pd.read_csv(os.path.join(TABLES, name)).set_index("label", drop=False)
     for name in os.listdir(TABLES) if name.endswith(".csv") and name not in
     ("key_comparisons.csv", "activation_search_choices.csv")}
KEY = pd.read_csv(os.path.join(TABLES, "key_comparisons.csv"))
CHOICES = pd.read_csv(os.path.join(TABLES, "activation_search_choices.csv"))
TOTAL_RUNS = sum(len(frame) for frame in RUNS.values())


# ---------------------------------------------------------------- formatting
def fmt2(value):
    """Two decimals, halves rounded up (98.625 -> 98.63)."""
    return str(Decimal(repr(round(float(value), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def two_significant(value):
    """12.34 -> '12', 0.0456 -> '0.046', 112.7 -> '110', 284668 -> '280,000'."""
    if value == 0:
        return "0"
    rounded = float(f"{value:.2g}")
    return f"{rounded:,.0f}" if rounded >= 10 else f"{rounded:g}"


# ---------------------------------------------------------------- look-ups
def column(experiment, label, name):
    frame = RUNS[experiment]
    values = frame.loc[frame["label"] == label, name].to_numpy()
    if len(values) == 0:
        raise KeyError(f"{experiment}: no runs labelled {label!r}")
    return values


def mean(experiment, label, name="test_accuracy"):
    return 100 * column(experiment, label, name).mean()


def acc(experiment, label):
    return fmt2(mean(experiment, label))


def pm(experiment, label, name="test_accuracy"):
    values = 100 * column(experiment, label, name)
    return f"{fmt2(values.mean())} ± {fmt2(values.std(ddof=1))}"


def cell(experiment, label, name="test_accuracy"):
    """Table cell: mean with the standard deviation in brackets."""
    values = 100 * column(experiment, label, name)
    return f"{fmt2(values.mean())} ({fmt2(values.std(ddof=1))})"


def train_cell(experiment, label):
    return cell(experiment, label, "train_accuracy")


def diversity(experiment, label):
    return two_significant(float(np.median(column(experiment, label, "final_diversity"))))


def rms(experiment, label):
    return two_significant(float(np.median(column(experiment, label, "weight_norm"))))


def p_text(value):
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}" if round(value, 3) < 0.1 else f"p = {value:.2f}"


def p(experiment, a, b):
    """Uncorrected two-sided Mann-Whitney U test from key_comparisons.csv."""
    rows = KEY[(KEY["experiment"] == experiment) & (KEY["A"] == a) & (KEY["B"] == b)]
    if len(rows) != 1:
        raise KeyError(f"no comparison {experiment}: {a} vs {b}")
    return p_text(rows.iloc[0]["p_value"])


def kruskal(experiment, labels):
    """Kruskal-Wallis test on test accuracy across the given settings."""
    return stats.kruskal(*[column(experiment, label, "test_accuracy") for label in labels]).pvalue


def perfect_runs(label):
    return int((column("final", label, "test_accuracy") == 1.0).sum())


def chosen(pso, budget, architecture):
    part = CHOICES[(CHOICES["pso"] == pso) & (CHOICES["budget"] == budget)
                   & (CHOICES["architecture"] == architecture)]
    return dict(zip(part["chosen_activations"], part["runs"]))


def count_distinct_runs():
    """Runs with a different (configuration, seed) pair. Baseline settings
    appear in several experiments with the same seeds; those repeats are
    counted once. Cross-validation folds and control-function runs are all
    distinct."""
    from experiments.final import BUDGET_MULTIPLIERS, FINAL_CONFIGS, _scaled

    seen = set()
    for name in os.listdir(RAW):
        if not name.endswith("_settings.json"):
            continue
        with open(os.path.join(RAW, name)) as handle:
            settings = json.load(handle)
        frame = RUNS[name.replace("_settings.json", "")]
        for label, seed in zip(frame["label"], frame["seed"]):
            seen.add((json.dumps(settings[label], sort_keys=True), int(seed)))
    for config in FINAL_CONFIGS.values():
        for multiplier in BUDGET_MULTIPLIERS:
            for seed in range(30):
                seen.add((json.dumps(_scaled(config, multiplier), sort_keys=True), seed))
    return len(seen) + len(RUNS["final_cv"]) + len(RUNS["landscape"])


def count_unit_tests():
    output = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                            cwd=ROOT, capture_output=True, text=True).stdout
    return int(re.search(r"(\d+) tests? collected", output).group(1))


DISTINCT_RUNS = count_distinct_runs()
UNIT_TESTS = count_unit_tests()

# Numbers taken from the per-iteration curves of the budget experiment.
_curves = np.load(os.path.join(RAW, "budget_curves.npz"))
LAST_IMPROVEMENT_5 = float(np.median([
    np.flatnonzero(np.diff(run) > 0).max() + 2 if (np.diff(run) > 0).any() else 1
    for run in _curves["tuned | budget 2000 | 5x400::fitness_history"]]))

# Facts about the coefficient grid that the text relies on.
_grid = T["coefficients"]
BEST_TRAIN_WHILE_SPREADING = _grid.loc[_grid["final_diversity_median"] > 10, "train_accuracy_mean"].max()
assert BEST_TRAIN_WHILE_SPREADING < 99.4
_worst = _grid.sort_values("train_accuracy_mean").head(3)
assert set(_worst["phi"]) == {0.5} and (_worst["final_diversity_median"] < 1).all()
assert _grid["train_accuracy_mean"].idxmax() == "alpha 0.4 | phi 2.0"


# ------------------------------------------------------------------ blocks
def h1(text):
    return {"type": "h1", "text": text}


def para(text):
    return {"type": "p", "text": " ".join(text.split())}


def figure(name, caption, width_in=7.1):
    return {"type": "figure", "file": os.path.join(FIGURES, name), "caption": " ".join(caption.split()),
            "width_in": width_in}


def table(caption, columns, rows, font_size=12):
    return {"type": "table", "caption": " ".join(caption.split()), "columns": columns, "rows": rows,
            "font_size": font_size}


def col(header, width_in, align="left"):
    return {"header": header, "width_in": width_in, "align": align}


# ------------------------------------------------------- frequently used labels
STD, TUN = "standard", "tuned"
BASE_T = "tuned | 4-8-1"
ARCHITECTURES = ["4-1", "4-2-1", "4-4-1", "4-8-1", "4-16-1", "4-32-1",
                 "4-4-4-1", "4-8-8-1", "4-16-16-1", "4-8-8-8-1"]
ACTIVATION_NAMES = {"logistic": "logistic", "tanh": "tanh", "relu": "ReLU",
                    "leaky_relu": "leaky ReLU", "elu": "ELU", "linear": "linear"}
FINAL_1K, FINAL_5K = "final | 1000 evals", "final | 5000 evals"

picked = chosen(TUN, 1000, "4-8-1")
picked_standard = chosen(STD, 1000, "4-8-1")
assert sum(picked.values()) == 30 and picked.get("logistic", 0) == 0
cv_runs = len(column("final_cv", FINAL_1K, "test_accuracy"))
walls_one = [f"tuned | {handler} +/-1" for handler in ("clamp", "reflect", "wrap", "random")]
mixed_standard = [mean("balance", f"standard | beta {b:g} gamma {3 - b:g}") for b in (0.5, 1, 1.5, 2, 2.5)]

# ============================================================== MAIN BODY
body = []

# ---------------------------------------------------------------- Section 1
body.append(h1("1  Implementation"))
body.append(para("""
**1.1 Overview.** We implemented a feedforward ANN and PSO in Python with NumPy as the only
numerical library. As in the design guidelines given in class, a Builder assembles the network, informant
selection is a Strategy, and an Adapter (Gamma *et al.*, 1994) is the only place where
the PSO and ANN packages meet (Table 1).
"""))
body.append(table(
    "**Table 1.** Structure of the code.",
    [col("Files", 2.25), col("Classes and functions", 5.0)],
    [
        ["`src/mlp/activations.py`",
         "One class per activation with `forward(z)`: logistic, tanh, ReLU; also leaky ReLU, ELU, "
         "linear, softmax."],
        ["`src/mlp/layer.py`, `network.py`, `builder.py`, `losses.py`",
         "`Layer`: weights, biases, activation. `Network`: a list of layers with `forward(X)`, and "
         "`get_parameters()` / `set_parameters(v)` to read or write all weights and biases as one "
         "vector. `MLPBuilder` builds a network from lists of layer sizes and activations, so depth, "
         "width and each layer's activation are configurable."],
        ["`src/pso/particle.py`, `population.py`, `fitness_function.py`",
         "`Particle`: position, velocity, personal best, informants; `update_velocity()`, "
         "`update_position()`. `Population`: creates the swarm; main loop in `optimise()`. "
         "`FitnessFunction`: the interface PSO optimises."],
        ["`src/pso/informant_strategy.py`, `boundary.py`",
         "Informant selection (random, nearest neighbours, ring) and boundary handling (none, "
         "clamp, reflect, wrap, random)."],
        ["`src/adapter/`",
         "`MLPFitnessAdapter.evaluate(position)` scores a particle as a network; "
         "`ActivationSearchAdapter` also decodes activation genes."],
        ["`src/data/`, `src/evaluation/`, `main.py`, `run_experiments.py`, `analyse_results.py`, "
         "`experiments/`, `tests/`",
         f"Stratified split, standardisation and metrics (our own NumPy code); train and test one "
         f"network; run the experiments; tables, tests and figures; {UNIT_TESTS} unit tests."],
    ],
))
body.append(para("""
**1.2 PSO.** `Population.optimise()` follows Algorithm 39 of Luke (2013), and comments such as
`[A39 L24]` give the line of the algorithm each statement implements. Every iteration assesses
each particle (lines 12–13), updates the best position found (14–15), computes all new
velocities from each particle's own best x{ast}, its informants' best x^{+} and the global best
x^{!}, with separate random numbers per dimension (16–24), and only then moves the particles
(25–26). Here α is the proportion of velocity retained, β, γ and δ weight the
pulls towards x{ast}, x^{+} and x^{!}, and ε is the jump size. Where the algorithm leaves a choice we
followed Luke's text: fitness is maximised; each particle has k informants, re-drawn at random
every iteration, and counts among its own informants; initial velocities are half the vector
between two random points.
"""))
body.append(para("""
**1.3 Coupling PSO to the ANN.** A particle's position is a vector of D real numbers, one per
weight and bias: D = Σ_{l} (n_{l−1} n_{l} + n_{l}) for layer sizes n_{l}, so a 4-8-1 network has
D = 49. PSO only sees an object with one method, `evaluate(position)`, which
`MLPFitnessAdapter` implements in three steps. `Network.set_parameters()` cuts the vector into
each layer's weight matrix and bias vector in a fixed order (layer by layer, weights then
biases). The network classifies the training set, thresholding its single logistic output at
0.5. The fraction classified correctly is returned as the particle's fitness. The architecture,
and so the vector length, is fixed for a run. After PSO stops, the best position is written into
the network and the test set is used for the first and only time.
"""))
body.append(para("""
**1.4 Beyond the specification.** We added (i) four boundary handling techniques and a velocity
limit; (ii) nearest-neighbour and ring informants; (iii) a linearly
decreasing inertia weight (Shi and Eberhart, 1998); (iv) cross-entropy and mean squared error
(MSE) as alternative fitness measures; and (v) PSO-chosen activation functions. For (v) the
particle gets one extra coordinate per hidden layer, whose range is cut into equal bins, one per
candidate activation (values outside it count as the nearest end bin); the adapter sets the
layer's activation from the bin before installing the weights. The swarm also records its
*diversity*, the mean distance of the particles from their centroid, our most useful diagnostic.
"""))
body.append(para("""
**1.5 Development and use of generative AI.** Our first version implemented the class design on
the Iris data and minimised a loss; it already had nearest-neighbour informants and the
global-best term. To bring it in line with this specification we used a generative AI assistant
(Claude, by Anthropic, through Claude Code). It reordered the PSO loop to match Algorithm 39 and
added the line-number comments; added ReLU and three further activations; made accuracy the
fitness; replaced Iris with the banknote data, and scikit-learn's split and metrics with NumPy
code; added boundary handling, the velocity limit, ring informants, the inertia schedule and the
activation search; wrote the experiment scripts and the tests; ran the experiments; and drafted
this report. Appendix A gives details and the prompt.
"""))

# ---------------------------------------------------------------- Section 2
body.append(h1("2  Results and Discussion"))
body.append(para("""
**2.1 Questions and method.** We address the three required questions and four more: informants, boundary
handling, the fitness measure and PSO-chosen activations. The data (Lohweg, 2012) has 1372
samples; a fixed stratified 70/30 split gives 960 training and 412 test samples, with inputs
standardised using the training set's statistics.
"""))
body.append(para("""
Each experiment varies one or two hyperparameters around a baseline from the literature: a 4-8-1
tanh network; α = 0.72 and β = γ = 1.5, close to the constriction values of Clerc and Kennedy
(2002); δ = 0 and ε = 1, as Luke (2013) suggests; 25 particles (about 10 + 2√D) with 3 random
informants, as in Standard PSO (Clerc, 2012); and 40 iterations, i.e. 1000 evaluations.
Particles start uniformly in [−1, 1]^{D} with no walls.
"""))
body.append(para(f"""
Every setting is run 30 times. We report means and standard deviations of accuracy and compare
settings with two-sided Mann–Whitney U tests (Derrac *et al.*, 2011); p-values in the text are
single pairwise tests, and those in Appendix B are Holm-corrected. To keep the test set out of
model selection (Lones, 2021), 'best' always means the highest mean *training* accuracy. The
study comprises {DISTINCT_RUNS:,} distinct runs.
"""))
body.append(para("""
With 25 particles the baseline coefficients suit this problem poorly (Section 2.4), so we took
the best cell of the coefficient grid, α = 0.4 with β = γ = 1, as a second set and ran every
experiment under both, called *standard* and *tuned* below. Several conclusions reverse between
them.
"""))

body.append(para(f"""
**2.2 Q1: Architecture.** With standard coefficients (Table 2) the network with no hidden layer
is nominally the best, and accuracy falls as parameters are added, to
{acc('architecture', 'standard | 4-8-8-8-1')}% with three hidden layers. With tuned coefficients
the trend reverses: accuracy rises from {acc('architecture', 'tuned | 4-1')}% with no hidden layer
to {acc('architecture', 'tuned | 4-4-1')}% with 4 hidden neurons and {acc('architecture', BASE_T)}%
with 8 ({p('architecture', BASE_T, 'tuned | 4-1')}), and more width costs nothing.
Depth does cost: with the same D = 193, 4-32-1 beats 4-8-8-8-1 under both sets
({p('architecture', 'tuned | 4-32-1', 'tuned | 4-8-8-8-1')}).
"""))

# Table 2: architecture (left) and activation (right)
right_rows = [(f"4-8-1, {ACTIVATION_NAMES[a]}", f"standard | 4-8-1 | {a}", f"tuned | 4-8-1 | {a}")
              for a in ["logistic", "tanh", "relu", "leaky_relu", "elu", "linear"]]
right_rows += [(f"4-8-8-1, {ACTIVATION_NAMES[a]}", f"standard | 4-8-8-1 | {a}", f"tuned | 4-8-8-1 | {a}")
               for a in ["logistic", "linear"]]
table2_rows = []
for index, arch in enumerate(ARCHITECTURES):
    row = [arch, str(int(column("architecture", f"standard | {arch}", "dimension")[0])),
           cell("architecture", f"standard | {arch}"), cell("architecture", f"tuned | {arch}")]
    if index < len(right_rows):
        name, std_label, tun_label = right_rows[index]
        row += [name, cell("activation", std_label), cell("activation", tun_label)]
    else:
        row += ["", "", ""]
    table2_rows.append(row)
body.append(table(
    """**Table 2.** Test accuracy (%, mean and s.d. of 30 runs) by architecture (tanh hidden units) and
    by hidden activation.""",
    [col("Network", 0.95), col("D", 0.42, "right"), col("Standard", 1.12, "right"),
     col("Tuned", 1.12, "right"), col("  Network, activation", 1.6), col("Standard", 1.02, "right"),
     col("Tuned", 1.02, "right")],
    table2_rows,
))
body.append(para("""
The classes are almost linearly separable (one logistic unit reaches about 98.5%), so capacity
is worth about one point and a small hidden layer collects it. Depth adds none that is needed,
and we think it makes the search harder because each weight acts through more non-linearities.
When the swarm is not converging extra width hurts too, and 'simpler is better' would be a
conclusion about the optimiser, not the network.
"""))
body.append(para(f"""
Among activations (tuned), ReLU and tanh are indistinguishable
({p('activation', 'tuned | 4-8-1 | relu', 'tuned | 4-8-1 | tanh')}), and logistic is worse
({p('activation', 'tuned | 4-8-1 | tanh', 'tuned | 4-8-1 | logistic')}), falling to
{acc('activation', 'tuned | 4-8-8-1 | logistic')}% with two hidden layers. We suspect a plateau:
logistic outputs are all positive with slope at most 0.25, so a second layer sees almost
constant inputs. Under standard coefficients linear hidden layers are best
({p('activation', 'standard | 4-8-8-1 | linear', 'standard | 4-8-8-1 | tanh')} with two layers):
the non-linearity is unused.
"""))

body.append(para(f"""
**2.3 Q2: Allocating evaluations.** For budgets of 500, 1000 and 2000 evaluations we varied the
swarm from 5 to 250 particles (Fig. 1). With standard
coefficients the fewest particles are best at every budget: at 500 evaluations, 5 × 100
(particles × iterations) reaches
{acc('budget', 'standard | budget 500 | 5x100')}%, against
{acc('budget', 'standard | budget 500 | 250x2')}% for 250 × 2. With tuned coefficients five
particles stall at {acc('budget', 'tuned | budget 500 | 5x100')}%,
with identical results at all three budgets. The best size is 25 at every budget; 10 particles
tie with it at 500 evaluations ({p('budget', 'tuned | budget 500 | 25x20', 'tuned | budget 500 | 10x50')})
and 50 at 2000 ({p('budget', 'tuned | budget 2000 | 25x80', 'tuned | budget 2000 | 50x40')}), so
the range of good sizes moves up as the budget grows.
"""))
body.append(figure("fig_budget.png", """
**Fig. 1.** Mean test accuracy against swarm size for three evaluation budgets (bars: 95%
confidence intervals).
"""))
body.append(para(f"""
With standard coefficients a swarm of 5 does not expand (final diversity {diversity('budget', 'standard | budget 2000 | 5x400')}–{diversity('budget', 'standard | budget 500 | 5x100')},
against 30 or more for 10 and 25 particles) and does well; with tuned coefficients half the
runs have stopped improving by iteration {LAST_IMPROVEMENT_5:.0f}. Overall, the best standard
allocation at 500 evaluations (5 × 100) and the best tuned one (25 × 20,
{acc('budget', 'tuned | budget 500 | 25x20')}%) do not differ significantly
({p('budget', 'standard | budget 500 | 5x100', 'tuned | budget 500 | 25x20')}): a small swarm is
another way of making the standard coefficients converge. Piotrowski *et al.* (2020) also found
the best swarm size to be problem-dependent.
"""))

body.append(para(f"""
**2.4 Q3: Velocity coefficients.** A 7 × 8 grid over α and β + γ, with β = γ (Fig. 2a), shows
that accuracy is highest along a diagonal band: less inertia can be traded for more attraction.
The best cell, α = 0.4 with β = γ = 1, reaches {pm('coefficients', 'alpha 0.4 | phi 2.0')}% test
accuracy against {pm('coefficients', 'alpha 0.72 | phi 3.0')}% for the standard values
({p('coefficients', 'alpha 0.4 | phi 2.0', 'alpha 0.72 | phi 3.0')}).
"""))
body.append(para(f"""
Swarm diversity explains the upper edge of the band (Fig. 2c). All swarms start at a diversity
of 3.9, and no setting that ended above 10 reached 99.4% training accuracy. Contracting is
necessary but not sufficient: the three worst cells, all with β + γ = 0.5, also ended below 1.
The standard setting ends at
{diversity('coefficients', 'alpha 0.72 | phi 3.0')}: its swarm *expands* throughout the run
(Fig. 3b), although it lies inside the region where theory says a particle's variance stays
bounded (Poli, 2009).
"""))
body.append(para(f"""
That theory assumes fixed attractors. Here they move, we think for two reasons. Random
informants re-drawn every iteration keep changing a particle's target (with nearest-neighbour
informants the same swarm ends at {diversity('informants', 'standard | nearest k=3')}). And
thresholded accuracy should change little when all weights are scaled up, so nothing pulls
particles back. A control experiment is consistent with this (Fig. 3c): on a function that
depends only on the direction of x the standard swarm's diversity grows from 3.9 to
{diversity('landscape', 'direction | standard')}, against
{diversity('landscape', 'sphere | standard')} on a sphere, while the tuned swarm contracts on
both. Particles leaving the initial region is a known effect in high dimensions (Helwig and
Wanka, 2008), and large weights saturate the hidden units of PSO-trained networks (Rakitianskaia
and Engelbrecht, 2015): the best particle's root-mean-square (r.m.s.) weight is {rms('coefficients', 'alpha 0.72 | phi 3.0')} for standard and
{rms('coefficients', 'alpha 0.4 | phi 2.0')} for tuned.
"""))
body.append(figure("fig_coefficients.png", """
**Fig. 2.** Velocity coefficients. (a) Mean test accuracy over α and β + γ; cells right of the
black line violate the stability condition β + γ < 24(1 − α^{2}) / (7 − 5α) (Poli, 2009). (b) A fixed
β + γ (3 for standard, 2 for tuned) split between cognitive and social pull (bars: 95% confidence
intervals). (c) Mean training accuracy of the 56 grid settings against median final diversity.
"""))
body.append(figure("fig_convergence.png", """
**Fig. 3.** The baseline network under both coefficient sets. (a) Best training accuracy (mean;
band: interquartile range). (b) Median swarm diversity. (c) Median diversity on a control
function that depends only on the direction of x.
"""))
body.append(para(f"""
*Cognitive against social* (Fig. 2b). With β + γ fixed, the tuned swarm does best with an even or
slightly social split. A purely social swarm (β = 0) loses
{mean('balance', 'tuned | beta 1 gamma 1') - mean('balance', 'tuned | beta 0 gamma 2'):.1f} points,
and a purely cognitive one (γ = 0) is the worst setting in the study
({pm('balance', 'tuned | beta 2 gamma 0')}%): nothing is shared, and with little inertia each
particle stalls near its own best, Luke's (2013) 'separate hill-climbers'. With standard
coefficients the mixed splits differ little ({fmt2(min(mixed_standard))}–{fmt2(max(mixed_standard))}%),
but final diversity falls from {diversity('balance', 'standard | beta 0 gamma 3')} to
{diversity('balance', 'standard | beta 3 gamma 0')} as the cognitive share rises: the social term
drives the expansion.
"""))
body.append(para(f"""
*Global best and jump size.* Moving half of the social pull onto the global best lifts the
standard swarm to {acc('global_best', 'standard | gamma 0.75 delta 0.75')}% but lowers the tuned
swarm to {acc('global_best', 'tuned | gamma 0.5 delta 0.5')}%, and halving ε does the same. A
linearly decreasing α (0.9 → 0.4) was no better than a constant
0.4 ({p('inertia_schedule', 'phi 2 | alpha constant 0.4', 'phi 2 | alpha 0.9 -> 0.4')}).
"""))

body.append(para(f"""
**2.5 Further experiments.** *Informants* (Fig. 4). With standard coefficients accuracy rises
with the number of informants, and nearest-neighbour or ring informants beat random ones
({acc('informants', 'standard | nearest k=3')}% for nearest against
{acc('informants', 'standard | random k=3')}% at k = 3,
{p('informants', 'standard | nearest k=3', 'standard | random k=3')}), because particles that
listen to nearby particles pull together. With tuned coefficients random informants with
k = 3 to 8 are best and nearest neighbours are worst
({acc('informants', 'tuned | nearest k=3')}% at k = 3), since neighbours hold nearly the same
information and the swarm splits into groups that converge separately. Topology is known to
matter (Kennedy and Mendes, 2002); here the best one depended on the coefficients.
"""))
body.append(figure("fig_informants.png", """
**Fig. 4.** Mean test accuracy against the number of informants for three selection rules (bars:
95% confidence intervals).
"""))

table3_rows = [
    ["None", cell("boundary", "standard | none"), cell("boundary", "tuned | none")],
    ["Walls at ±1, clamp", cell("boundary", "standard | clamp +/-1"), cell("boundary", "tuned | clamp +/-1")],
    ["Walls at ±1, reflect", cell("boundary", "standard | reflect +/-1"), cell("boundary", "tuned | reflect +/-1")],
    ["Walls at ±1, wrap", cell("boundary", "standard | wrap +/-1"), cell("boundary", "tuned | wrap +/-1")],
    ["Walls at ±1, random", cell("boundary", "standard | random +/-1"), cell("boundary", "tuned | random +/-1")],
    ["Velocity limit ±0.5", cell("velocity", "standard | vmax 0.5"), cell("velocity", "tuned | vmax 0.5")],
    ["Cross-entropy fitness", cell("fitness_function", "standard | 1000 evals | cross-entropy"),
     cell("fitness_function", "tuned | 1000 evals | cross-entropy")],
    ["MSE fitness", cell("fitness_function", "standard | 1000 evals | MSE"),
     cell("fitness_function", "tuned | 1000 evals | MSE")],
    ["PSO-chosen activation", cell("activation_search", "standard | 1000 evals | 4-8-1 | PSO-chosen"),
     cell("activation_search", "tuned | 1000 evals | 4-8-1 | PSO-chosen")],
]
body.append(table(
    """**Table 3.** Further experiments on the baseline network: test accuracy (%, mean and s.d. of
    30 runs).""",
    [col("Change to the baseline", 4.05), col("Standard", 1.6, "right"), col("Tuned", 1.6, "right")],
    table3_rows,
))
body.append(para(f"""
*Boundary handling* (Table 3). Walls matter only if particles reach them. The tuned swarm never
touches walls at ±5 or ±10, and at ±1 the four handlers do not differ (Kruskal–Wallis
p = {kruskal('boundary', walls_one):.2f}). For the standard swarm, clamping is best and wrapping
much the worst ({p('boundary', 'standard | clamp +/-1', 'standard | wrap +/-1')}): wrapping
carries a weight that passes +1 round to −1, flipping its sign, whereas clamping leaves it at
the nearest legal value. Helwig *et al.* (2013) also report large differences between bound
handling methods.
"""))
body.append(para(f"""
*Fitness measure.* Replacing accuracy with cross-entropy or MSE costs the tuned swarm about two
points ({p('fitness_function', 'tuned | 1000 evals | accuracy', 'tuned | 1000 evals | cross-entropy')}).
Tentatively, a loss can always be lowered by making already-correct outputs more confident
(r.m.s. weight {rms('fitness_function', 'tuned | 1000 evals | cross-entropy')} against
{rms('fitness_function', 'tuned | 1000 evals | accuracy')}), so the swarm converges before the
last errors are fixed.
"""))
body.append(para(f"""
*PSO-chosen activations.* Letting PSO choose the hidden activation costs a little at 1000
evaluations
({p('activation_search', 'tuned | 1000 evals | 4-8-1 | fixed tanh', 'tuned | 1000 evals | 4-8-1 | PSO-chosen')})
and nothing at 3000. Over 30 tuned runs it chose tanh {picked.get('tanh', 0)} times, ELU
{picked.get('elu', 0)}, leaky ReLU {picked.get('leaky_relu', 0)}, ReLU {picked.get('relu', 0)} and
logistic never, so it avoids the one clearly bad option. (With standard coefficients the genes
drift out of range and the end bins dominate: ELU {picked_standard.get('elu', 0)} times, logistic
{picked_standard.get('logistic', 0)}.)
"""))

body.append(para(f"""
**2.6 Final configuration and limitations.** No experiment found a setting significantly better
than the baseline network and swarm with tuned coefficients, so that is our final configuration
(Table 4). In 1000 evaluations it averages {acc('final', FINAL_1K)}% test accuracy, and
{perfect_runs(FINAL_1K)} of 30 runs classify all 412 test notes correctly. Five times as many
evaluations bring no significant gain ({acc('final', FINAL_5K)}%, {p('final', FINAL_5K, FINAL_1K)}).
Repeated stratified 10-fold cross-validation gave {pm('final_cv', FINAL_1K)}%. Kumar and Dudyala (2015) found multilayer perceptrons among the most effective
classifiers for this dataset; a 49-parameter network trained without gradients classifies it
almost perfectly.
"""))

table4_rows = []
for name in ("baseline", "final"):
    label = f"{name} | 1000 evals"
    table4_rows.append([
        "Standard (baseline)" if name == "baseline" else "Tuned (final)",
        train_cell("final", label), cell("final", label),
        fmt2(mean("final", label, "test_precision")), fmt2(mean("final", label, "test_recall")),
        fmt2(mean("final", label, "test_f1")), cell("final_cv", label),
    ])
body.append(table(
    f"""**Table 4.** Baseline and final configurations at 1000 evaluations: mean and s.d. (%) of 30 runs on the fixed
    split, and accuracy over {cv_runs} folds of repeated 10-fold cross-validation (CV).""",
    [col("Coefficients", 1.75), col("Train acc.", 1.15, "right"), col("Test acc.", 1.15, "right"),
     col("Precision", 0.75, "right"), col("Recall", 0.65, "right"), col("F1", 0.65, "right"),
     col("CV acc.", 1.15, "right")],
    table4_rows,
))
body.append(para("""
Limitations: the one dataset is nearly linearly separable, and differences among the good
settings (99.3–99.6%) are within noise. Hyperparameters were varied one or two at a time around
two points, so interactions may have been missed. The tuned coefficients were chosen with 25
particles, 3 random informants and the 4-8-1 network, on the same 30 seeds, which favours those
values. Settings were chosen by training accuracy, not on a validation
set.
"""))

# ---------------------------------------------------------------- Section 3
body.append(h1("3  Conclusions"))
body.append(para("""
We built a configurable ANN and the informant-based PSO of Algorithm 39 from scratch, coupled
them through a fitness adapter and trained the network to authenticate banknotes. Three
conclusions follow.
"""))
body.append(para(f"""
*The velocity coefficients matter most.* With 25 particles the usual values made the swarm
expand on this problem; α = 0.4 with β = γ = 1 raised mean test accuracy from
{mean('final', 'baseline | 1000 evals'):.1f}% to {mean('final', FINAL_1K):.1f}%. Swarms that kept
spreading never reached the best accuracies, though contracting was not sufficient.
"""))
body.append(para("""
*The other answers depended on the coefficients.* With a swarm that did not converge, the
smallest network, the smallest swarm, and many or local informants looked best. With tuned
coefficients each of these reversed, in favour of one small hidden layer, about 25 particles and
a few random informants.
"""))
body.append(para("""
*The extensions were of mixed value.* Boundary handling mattered only when particles reached the
walls, accuracy was a better fitness than a loss, and PSO-chosen activations gained nothing.
"""))
body.append(para("""
With more time we would adapt the inertia weight to the measured diversity, repeat
the study on a harder, non-separable problem where capacity and overfitting (van Wyk and
Engelbrecht, 2010) should matter more, and compare against backpropagation at equal cost.
"""))

# ============================================================== REFERENCES
references = [
    "Clerc, M. (2012) *Standard Particle Swarm Optimisation*. HAL open archive, hal-00764996. "
    "Available at: https://hal.science/hal-00764996 (Accessed: 27 November 2026).",
    "Clerc, M. and Kennedy, J. (2002) 'The particle swarm: explosion, stability, and convergence in "
    "a multidimensional complex space', *IEEE Transactions on Evolutionary Computation*, 6(1), pp. 58–73.",
    "Derrac, J., García, S., Molina, D. and Herrera, F. (2011) 'A practical tutorial on the use of "
    "nonparametric statistical tests as a methodology for comparing evolutionary and swarm "
    "intelligence algorithms', *Swarm and Evolutionary Computation*, 1(1), pp. 3–18.",
    "Gamma, E., Helm, R., Johnson, R. and Vlissides, J. (1994) *Design Patterns: Elements of Reusable "
    "Object-Oriented Software*. Reading, MA: Addison-Wesley.",
    "Helwig, S., Branke, J. and Mostaghim, S. (2013) 'Experimental analysis of bound handling "
    "techniques in particle swarm optimization', *IEEE Transactions on Evolutionary Computation*, "
    "17(2), pp. 259–271.",
    "Helwig, S. and Wanka, R. (2008) 'Theoretical analysis of initial particle swarm behavior', in "
    "*Parallel Problem Solving from Nature – PPSN X*. Lecture Notes in Computer Science, vol. 5199. "
    "Berlin: Springer, pp. 889–898.",
    "Kennedy, J. and Mendes, R. (2002) 'Population structure and particle swarm performance', in "
    "*Proceedings of the 2002 Congress on Evolutionary Computation (CEC'02)*. IEEE, pp. 1671–1676.",
    "Kumar, C. and Dudyala, A.K. (2015) 'Bank note authentication using decision tree rules and "
    "machine learning techniques', in *2015 International Conference on Advances in Computer "
    "Engineering and Applications*. IEEE. doi: 10.1109/ICACEA.2015.7164721.",
    "Lohweg, V. (2012) *Banknote Authentication* [Dataset]. UCI Machine Learning Repository. "
    "doi: 10.24432/C55P57.",
    "Lones, M.A. (2021) 'How to avoid machine learning pitfalls: a guide for academic researchers', "
    "*arXiv preprint* arXiv:2108.02497.",
    "Luke, S. (2013) *Essentials of Metaheuristics*. 2nd edn. Lulu. Available at: "
    "http://people.cs.gmu.edu/~sean/book/metaheuristics/ (Accessed: 27 November 2026).",
    "Piotrowski, A.P., Napiorkowski, J.J. and Piotrowska, A.E. (2020) 'Population size in particle "
    "swarm optimization', *Swarm and Evolutionary Computation*, 58, 100718.",
    "Poli, R. (2009) 'Mean and variance of the sampling distribution of particle swarm optimizers "
    "during stagnation', *IEEE Transactions on Evolutionary Computation*, 13(4), pp. 712–721.",
    "Rakitianskaia, A. and Engelbrecht, A.P. (2015) 'Saturation in PSO neural network training: good "
    "or evil?', in *2015 IEEE Congress on Evolutionary Computation (CEC)*. IEEE, pp. 125–132.",
    "Shi, Y. and Eberhart, R. (1998) 'A modified particle swarm optimizer', in *1998 IEEE "
    "International Conference on Evolutionary Computation Proceedings*. IEEE, pp. 69–73.",
    "van Wyk, A.B. and Engelbrecht, A.P. (2010) 'Overfitting by PSO trained feedforward neural "
    "networks', in *IEEE Congress on Evolutionary Computation*. IEEE, pp. 1–8.",
]

# ============================================================== APPENDICES
appendix = []
appendix.append(h1("Appendix A  Use of generative AI"))
appendix.append(para("""
**System.** Claude (Anthropic), used through Claude Code, an assistant that can read and edit
files and run commands in a sandbox. Used in November 2026.
"""))
appendix.append(para("""
**What it was given.** The full coursework specification, the class design guidelines
(*Coursework Software Design Guidelines*), our existing code (ANN, PSO and adapter written to
that design and run on Iris) and our draft report.
"""))
appendix.append(para("""
**Prompt.** One instruction was given, after the text of the specification; the part that
describes the task is quoted here as typed. "give this and the attached CW_design which has the
implementation i did originally but forgot about the above requirements can you modify the code.
I need you to fix all of the gaps that are in the requirement that i missed. Modify the code and
the report to adhere to the requirements. […] Make sure to adhere to the format and structure
required by the coursework." The assistant then worked without further prompts, deciding itself
which experiments and extensions to carry out.
"""))
appendix.append(para("""
**What it produced.** (1) Code: the changes listed in Section 1.5, that is, the Algorithm 39
loop and its line-number comments, the added activations, accuracy as the fitness, the NumPy
data handling and metrics, the boundary handlers, the ring informant strategy, the velocity
limit and inertia schedule, the activation-search adapter, the command-line interface, the
experiment runner, the analysis script and the unit tests. (2) Experiments: the choice of
experiments and value ranges, and all the runs. (3) Report: the text, tables and figures.
"""))
appendix.append(para(f"""
**How it was checked.** The assistant compared the PSO loop with Algorithm 39 in the book's PDF,
downloaded the data file from the UCI repository and checked its size and class counts, ran the
{UNIT_TESTS} unit tests, and ran the complete set of experiments twice to confirm that all
{TOTAL_RUNS:,} run records were identical. The numbers in this report are inserted from the
result files by a script, not typed. Three further instances of the assistant then reviewed the
report section by section against the code and the raw results; they found no errors in the
tables but did find one mis-quoted test and several overstated explanations, which were
corrected. Each reference was checked against a publisher, repository or index record; page
numbers that could not be confirmed were left out.
"""))
appendix.append(para("""
**[To be completed by the team before submission: how the first version of the code was written,
and what each of us reviewed or changed in the AI-generated code and text.]**
"""))


def result_table(caption, experiment, labels, names, first_header, widths=(3.05, 0.5, 1.2, 1.2, 0.7, 0.6)):
    rows = []
    for label, name in zip(labels, names):
        r = T[experiment].loc[label]
        if bool(r["best_in_block"]):
            p_cell = "best"
        elif pd.isna(r["p_holm"]):
            p_cell = ""
        else:
            p_cell = "<0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
        rows.append([name, str(int(column(experiment, label, "dimension")[0])),
                     train_cell(experiment, label), cell(experiment, label),
                     diversity(experiment, label), p_cell])
    return table(caption, [col(first_header, widths[0]), col("D", widths[1], "right"),
                           col("Train acc.", widths[2], "right"), col("Test acc.", widths[3], "right"),
                           col("Div.", widths[4], "right"), col("p", widths[5], "right")], rows)


appendix.append(h1("Appendix B  Additional results"))
appendix.append(para("""
The tables below give training as well as test accuracy (%, mean and s.d. of 30 runs), the median
final swarm diversity (Div.) and the Holm-corrected p-value of a two-sided Mann–Whitney U test
on test accuracy against the best setting of the same block (the one with the highest mean
training accuracy, marked 'best'). Diversity grows with D, so it should only be compared between
settings with the same D. The complete results of all 16 experiments are in `results/summary.md`
and `results/tables/` in the code submission.
"""))
for pso in (STD, TUN):
    appendix.append(result_table(
        f"**Table B{1 if pso == STD else 2}.** Architecture, {pso} coefficients (tanh hidden units).",
        "architecture", [f"{pso} | {a}" for a in ARCHITECTURES], ARCHITECTURES, "Network"))
for index, pso in enumerate((STD, TUN), start=3):
    labels, names = [], []
    for budget in (500, 1000, 2000):
        for n in (5, 10, 25, 50, 100, 250):
            labels.append(f"{pso} | budget {budget} | {n}x{budget // n}")
            names.append(f"{budget}: {n} × {budget // n}")
    appendix.append(result_table(
        f"**Table B{index}.** Allocation of evaluations, {pso} coefficients (p within each budget).",
        "budget", labels, names, "Evaluations: particles × iterations"))
for index, pso in enumerate((STD, TUN), start=5):
    labels, names = [], []
    for strategy, text in (("random", "random"), ("nearest", "nearest neighbours"), ("ring", "ring")):
        for k in (1, 3, 8, 24):
            labels.append(f"{pso} | {strategy} k={k}")
            names.append(f"{text}, k = {k}")
    appendix.append(result_table(
        f"**Table B{index}.** Informants, {pso} coefficients (k = 1, 3, 8, 24 of the 7 values tested).",
        "informants", labels, names, "Informant selection"))
labels = ([f"standard | beta {b:g} gamma {3 - b:g}" for b in (0, 0.5, 1, 1.5, 2, 2.5, 3)]
          + [f"tuned | beta {b} gamma {g}" for b, g in (("0", "2"), ("0.3333", "1.6667"),
                                                         ("0.6667", "1.3333"), ("1", "1"),
                                                         ("1.3333", "0.6667"), ("1.6667", "0.3333"),
                                                         ("2", "0"))])
names = ([f"α = 0.72, β = {b:g}, γ = {3 - b:g}" for b in (0, 0.5, 1, 1.5, 2, 2.5, 3)]
         + [f"α = 0.4, β = {b}, γ = {g}" for b, g in (("0", "2"), ("0.33", "1.67"), ("0.67", "1.33"),
                                                      ("1", "1"), ("1.33", "0.67"), ("1.67", "0.33"),
                                                      ("2", "0"))])
appendix.append(result_table(
    "**Table B7.** Cognitive against social pull with β + γ fixed (p within each value of α).",
    "balance", labels, names, "Coefficients"))

appendix.append(h1("Appendix C  Running the code"))
appendix.append(para("""
The code needs Python 3.9 or newer and the packages in `requirements.txt` (NumPy for the ANN and
PSO; pandas, SciPy and matplotlib for the analysis only). From the top folder:
`python main.py` trains and tests one network with the final configuration, and
`python main.py --help` lists the options for the network and the swarm;
`python run_experiments.py --experiment all` repeats the whole study (about six minutes on four
cores) and `python analyse_results.py` rebuilds the tables and figures; `python -m pytest`
runs the tests. `README.md` gives more detail.
"""))
appendix.append(h1("Appendix D  Group signing sheet"))
appendix.append(para("""
**[Insert the completed and signed Coursework Group Signing Sheet from Canvas here, or include
it in the code zip file.]**
"""))

# ------------------------------------------------------------------ output
front = {
    "campus": "Dubai Campus, School of Mathematical and Computer Sciences",
    "course": "F21BC Biologically Inspired Computation, Coursework Report, November 2026",
    "title": "A Multilayer Perceptron Trained by Particle Swarm Optimisation",
    "authors": "Ahmed Abdallah (H00562295) and Mohamed Ali (H00562194)",
    "code": "Code: github.com/mr1pro/bio-inspired-coursework",
    "logo": os.path.join(HERE, "assets", "hw_logo.png"),
}


def plain(text):
    text = text.replace("{ast}", "*")
    return re.sub(r"[*`]|\^\{|_\{|\}", "", text)


def count_words(blocks):
    words = 0
    for block in blocks:
        if block["type"] in ("h1", "p"):
            words += len(plain(block["text"]).split())
        elif block["type"] == "figure":
            words += len(plain(block["caption"]).split())
        elif block["type"] == "table":
            words += len(plain(block["caption"]).split())
            words += sum(len(plain(c["header"]).split()) for c in block["columns"])
            words += sum(len(plain(cell).split()) for row in block["rows"] for cell in row)
    return words


if __name__ == "__main__":
    prose = sum(len(plain(b["text"]).split()) for b in body if b["type"] in ("h1", "p"))
    total = count_words(body) + sum(len(front[k].split()) for k in ("campus", "course", "title", "authors", "code"))
    with open(os.path.join(HERE, "content.json"), "w", encoding="utf-8") as handle:
        json.dump({"front": front, "body": body, "references": references, "appendix": appendix},
                  handle, ensure_ascii=False, indent=1)
    print(f"body prose and headings: {prose} words")
    print(f"body including title block, captions and table cells: {total} words (limit 3000)")
    print(f"references: {len(references)}   distinct runs: {DISTINCT_RUNS:,}   run records: {TOTAL_RUNS:,}"
          f"   unit tests: {UNIT_TESTS}")
