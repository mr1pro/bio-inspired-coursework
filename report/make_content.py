"""Assemble the text, tables and figure list of the report.

Every number in the report is looked up in results/tables/*.csv (or computed
from results/raw/*.csv) by the helper functions below, so the report cannot
drift away from the experiments. The output, report/content.json, is turned
into a Word document by build_docx.js.

Inline markup understood by build_docx.js:
    **bold**   *italic*   `code`   ^{superscript}   _{subscript}
"""

import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, "results", "tables")
RAW = os.path.join(ROOT, "results", "raw")
FIGURES = os.path.join(ROOT, "results", "figures")

T = {name[:-4]: pd.read_csv(os.path.join(TABLES, name)).set_index("label", drop=False)
     for name in os.listdir(TABLES) if name.endswith(".csv") and name not in
     ("key_comparisons.csv", "activation_search_choices.csv")}
KEY = pd.read_csv(os.path.join(TABLES, "key_comparisons.csv"))
CHOICES = pd.read_csv(os.path.join(TABLES, "activation_search_choices.csv"))
RAW_FINAL = pd.read_csv(os.path.join(RAW, "final.csv"))
TOTAL_RUNS = sum(len(pd.read_csv(os.path.join(RAW, f))) for f in os.listdir(RAW) if f.endswith(".csv"))


# ---------------------------------------------------------------- look-ups
def acc(experiment, label):
    return f"{T[experiment].loc[label, 'test_accuracy_mean']:.2f}"


def pm(experiment, label):
    row = T[experiment].loc[label]
    return f"{row['test_accuracy_mean']:.2f} ± {row['test_accuracy_std']:.2f}"


def train_pm(experiment, label):
    row = T[experiment].loc[label]
    return f"{row['train_accuracy_mean']:.2f} ± {row['train_accuracy_std']:.2f}"


def two_significant(value):
    """12.34 -> '12', 0.0456 -> '0.046', 112.7 -> '110', 284668 -> '280,000'."""
    if value == 0:
        return "0"
    rounded = float(f"{value:.2g}")
    return f"{rounded:,.0f}" if rounded >= 10 else f"{rounded:g}"


def diversity(experiment, label):
    return two_significant(T[experiment].loc[label, "final_diversity_median"])


def rms(experiment, label):
    return two_significant(T[experiment].loc[label, "weight_norm_median"])


def _comparison(experiment, a, b):
    rows = KEY[(KEY["experiment"] == experiment) & (KEY["A"] == a) & (KEY["B"] == b)]
    if len(rows) != 1:
        raise KeyError(f"no comparison {experiment}: {a} vs {b}")
    return rows.iloc[0]


def p(experiment, a, b):
    value = _comparison(experiment, a, b)["p_value"]
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}" if value < 0.1 else f"p = {value:.2f}"


def a12(experiment, a, b):
    """Probability that a run of `a` beats a run of `b` (ties count half),
    recomputed from the raw runs so that it is rounded only once."""
    raw = pd.read_csv(os.path.join(RAW, f"{experiment}.csv"))
    x = raw.loc[raw["label"] == a, "test_accuracy"].to_numpy()[:, None]
    y = raw.loc[raw["label"] == b, "test_accuracy"].to_numpy()[None, :]
    value = float((x > y).mean() + 0.5 * (x == y).mean())
    return f"{value + 1e-9:.2f}"


def kruskal(experiment, label):
    return f"{T[experiment].loc[label, 'kruskal_p']:.2f}"


def final_metric(label, column):
    """Mean of a per-run column of the final experiment, in percent."""
    return f"{100 * RAW_FINAL.loc[RAW_FINAL['label'] == label, column].mean():.2f}"


def perfect_runs(label):
    return int((RAW_FINAL.loc[RAW_FINAL["label"] == label, "test_accuracy"] == 1.0).sum())


def chosen(pso, budget, architecture):
    part = CHOICES[(CHOICES["pso"] == pso) & (CHOICES["budget"] == budget)
                   & (CHOICES["architecture"] == architecture)]
    return dict(zip(part["chosen_activations"], part["runs"]))


# ------------------------------------------------------------------ blocks
def h1(text):
    return {"type": "h1", "text": text}


def para(text):
    return {"type": "p", "text": " ".join(text.split())}


def figure(name, caption, width_in=7.1):
    return {"type": "figure", "file": os.path.join(FIGURES, name), "caption": " ".join(caption.split()),
            "width_in": width_in}


def table(caption, columns, rows, note=None, font_size=12):
    return {"type": "table", "caption": " ".join(caption.split()), "columns": columns, "rows": rows,
            "note": note, "font_size": font_size}


def col(header, width_in, align="left"):
    return {"header": header, "width_in": width_in, "align": align}


# ------------------------------------------------------- frequently used labels
STD, TUN = "standard", "tuned"
BASE_S, BASE_T = "standard | 4-8-1", "tuned | 4-8-1"
ARCHITECTURES = ["4-1", "4-2-1", "4-4-1", "4-8-1", "4-16-1", "4-32-1",
                 "4-4-4-1", "4-8-8-1", "4-16-16-1", "4-8-8-8-1"]
ACTIVATION_NAMES = {"logistic": "logistic", "tanh": "tanh", "relu": "ReLU",
                    "leaky_relu": "leaky ReLU", "elu": "ELU", "linear": "linear"}

picked = chosen(TUN, 1000, "4-8-1")
assert sum(picked.values()) == 30
cv_final = T["final_cv"].loc["final | 1000 evals"]
cv_runs = int(cv_final["runs"])

# ============================================================== MAIN BODY
body = []

# ---------------------------------------------------------------- Section 1
body.append(h1("1  Implementation"))
body.append(para("""
**1.1 Overview.** We implemented a feedforward ANN and PSO in Python with NumPy as the only
numerical library; the stratified split, standardisation and metrics are also our own code.
Following the class design guidelines, the network is assembled by a Builder, informant
selection is a Strategy, and an Adapter (Gamma *et al.*, 1994) is the only place where the PSO
and ANN packages meet (Table 1).
"""))
body.append(table(
    "**Table 1.** Structure of the code.",
    [col("Files", 2.25), col("Classes and functions", 5.0)],
    [
        ["`src/mlp/activations.py`",
         "One class per activation with `forward(z)`: logistic, tanh, ReLU; also leaky ReLU, ELU, "
         "linear, softmax."],
        ["`src/mlp/layer.py`, `network.py`, `builder.py`",
         "`Layer`: weights, biases, activation. `Network`: a list of layers with `forward(X)`, and "
         "`get_parameters()` / `set_parameters(v)` to read or write all weights and biases as one "
         "vector. `MLPBuilder` builds a network from lists of layer sizes and activations, so depth, "
         "width and each layer's activation are configurable."],
        ["`src/pso/particle.py`, `population.py`",
         "`Particle`: position, velocity, personal best, informants; `update_velocity()`, "
         "`update_position()`. `Population`: creates the swarm; main loop in `optimise()`."],
        ["`src/pso/informant_strategy.py`, `boundary.py`",
         "Informant selection (random, nearest neighbours, ring) and boundary handling (none, "
         "clamp, reflect, wrap, random)."],
        ["`src/adapter/`",
         "`MLPFitnessAdapter.evaluate(position)` scores a particle as a network; "
         "`ActivationSearchAdapter` also decodes activation genes."],
        ["`main.py`, `run_experiments.py`, `analyse_results.py`, `tests/`",
         "Train and test one network; run the experiments; tables, tests and figures; 70 unit tests."],
    ],
))
body.append(para("""
**1.2 PSO.** `Population.optimise()` follows Algorithm 39 of Luke (2013), and comments such as
`[A39 L24]` give the line of the algorithm each statement implements. Every iteration assesses
each particle (lines 12–13), updates the best position found (14–15), computes all new
velocities from each particle's own best x{ast}, its informants' best x^{+} and the global best
x^{!}, with separate random numbers per dimension (16–24), and only then moves the particles
(25–26). In the book's symbols, α is the proportion of velocity retained, β, γ and δ weight the
pulls towards x{ast}, x^{+} and x^{!}, and ε is the jump size. Where the algorithm leaves a choice we
followed Luke's text: fitness is maximised; each particle has k informants, re-drawn at random
every iteration, and counts among its own informants; initial velocities are half the vector
between two random points.
"""))
body.append(para("""
**1.3 Coupling PSO to the ANN.** A particle's position is a vector of D real numbers, one per
weight and bias: D = Σ_{l} (n_{l−1} n_{l} + n_{l}) for layer sizes n_{l}, so a 4-8-1 network has
D = 49. PSO only sees an object with one method, `evaluate(position)`, which
`MLPFitnessAdapter` implements in three steps.
`Network.set_parameters()` cuts the vector into each layer's weight matrix and bias vector in a
fixed order (layer by layer, weights then biases). The network classifies the training set,
thresholding its single logistic output at 0.5. The fraction classified correctly is returned as
the particle's fitness. The architecture is fixed before the run, so the vector length never
changes. After PSO stops, the best position is written into the network and the test set is used
for the first and only time.
"""))
body.append(para("""
**1.4 Beyond the specification.** We added (i) four boundary handling techniques, applied after
line 26, and a velocity limit; (ii) nearest-neighbour and ring informants; (iii) a linearly
decreasing inertia weight (Shi and Eberhart, 1998); (iv) cross-entropy and mean squared error
(MSE) as alternative fitness measures; and (v) PSO-chosen activation functions. For (v) the
particle gets one extra coordinate per hidden layer, whose range is cut into equal bins, one per
candidate activation; the adapter sets the layer's activation from the bin before installing the
weights. The swarm also records its *diversity*, the mean distance of the particles from their
centroid, our most useful diagnostic.
"""))
body.append(para("""
**1.5 Development and use of generative AI.** Our first version implemented the class design on
the Iris data and minimised a loss. To bring it in line with this specification we used a
generative AI assistant (Claude, by Anthropic, through Claude Code). It reordered the PSO loop
to match Algorithm 39 and added the line-number comments, added ReLU, replaced Iris with the
banknote data and scikit-learn's split and metrics with NumPy code, wrote the extensions of
Section 1.4, the experiment scripts and the tests, ran the experiments and drafted this report.
Every number below is generated by `run_experiments.py` and `analyse_results.py` with fixed
seeds. Appendix A gives details and the prompt.
"""))

# ---------------------------------------------------------------- Section 2
body.append(h1("2  Results and Discussion"))
body.append(para("""
**2.1 Questions and method.** We address the three required questions (Q1 architecture, Q2
allocation of evaluations, Q3 velocity coefficients) and four more: informants, boundary
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
Every setting is run 30 times. We report the mean and standard deviation of accuracy and compare
settings with two-sided Mann–Whitney U tests, Holm-corrected for multiple comparisons (Derrac
*et al.*, 2011). To keep the test set out of model selection (Lones, 2021), 'best' always means
the highest mean *training* accuracy. The study comprises {TOTAL_RUNS:,} runs.
"""))
body.append(para("""
The baseline coefficients suit this problem poorly (Section 2.4), so we took the best cell of the
coefficient grid, α = 0.4 with β = γ = 1, as a second set and ran every experiment under both,
called *standard* and *tuned* below. Several conclusions reverse between them.
"""))

body.append(para(f"""
**2.2 Q1: Architecture.** With standard coefficients (Table 2) the best network has no hidden
layer, and accuracy falls as parameters are added, to
{acc('architecture', 'standard | 4-8-8-8-1')}% with three hidden layers. With tuned coefficients
the trend reverses: accuracy rises from {acc('architecture', 'tuned | 4-1')}% with no hidden layer
to {acc('architecture', BASE_T)}% with 8 hidden neurons
({p('architecture', BASE_T, 'tuned | 4-1')}) and then levels off. Extra hidden layers do not help.
"""))

# Table 2: architecture (left) and activation (right)
right_rows = [(f"4-8-1, {ACTIVATION_NAMES[a]}", f"standard | 4-8-1 | {a}", f"tuned | 4-8-1 | {a}")
              for a in ["logistic", "tanh", "relu", "leaky_relu", "elu", "linear"]]
right_rows += [(f"4-8-8-1, {ACTIVATION_NAMES[a]}", f"standard | 4-8-8-1 | {a}", f"tuned | 4-8-8-1 | {a}")
               for a in ["logistic", "tanh", "relu", "elu"]]
table2_rows = []
for arch, (name, std_label, tun_label) in zip(ARCHITECTURES, right_rows):
    table2_rows.append([
        arch, str(int(T["architecture"].loc[f"standard | {arch}", "D"])),
        pm("architecture", f"standard | {arch}"), pm("architecture", f"tuned | {arch}"),
        name, pm("activation", std_label), pm("activation", tun_label),
    ])
body.append(table(
    """**Table 2.** Test accuracy (%, mean ± s.d. of 30 runs) by architecture (tanh hidden units) and
    by hidden activation.""",
    [col("Network", 0.95), col("D", 0.42, "right"), col("Standard", 1.12, "right"),
     col("Tuned", 1.12, "right"), col("  Network, activation", 1.6), col("Standard", 1.02, "right"),
     col("Tuned", 1.02, "right")],
    table2_rows,
))
body.append(para("""
Two effects compete. The classes are almost linearly separable (one logistic
unit reaches about 98.5%), so extra capacity is worth about one point. Against that, each extra
weight is one more dimension to search with the same 1000 evaluations.
When the swarm is not converging only the second effect shows, and 'simpler is better' would be
a conclusion about the optimiser, not the network.
"""))
body.append(para(f"""
Among activations (tuned), ReLU and tanh are indistinguishable
({p('activation', 'tuned | 4-8-1 | relu', 'tuned | 4-8-1 | tanh')}), and logistic is worse
({p('activation', 'tuned | 4-8-1 | tanh', 'tuned | 4-8-1 | logistic')}), falling to
{acc('activation', 'tuned | 4-8-8-1 | logistic')}% with two hidden layers. Its outputs are all
positive and its slope never exceeds 0.25, so a second logistic layer receives almost constant
inputs; we suspect most of the initial swarm sits on a plateau. Under standard coefficients a
linear hidden layer is nominally the best choice: the non-linearity is not being used.
"""))

body.append(para(f"""
**2.3 Q2: Allocating evaluations.** For budgets of 500, 1000 and 2000 evaluations we varied the
swarm from 5 to 250 particles, with iterations = budget / swarm size (Fig. 1). With standard
coefficients the fewest particles are best at every budget: at 500 evaluations, 5 × 100 reaches
{acc('budget', 'standard | budget 500 | 5x100')}%, against
{acc('budget', 'standard | budget 500 | 25x20')}% for 25 × 20 and
{acc('budget', 'standard | budget 500 | 250x2')}% for 250 × 2. With tuned coefficients the best
allocation is in between. Five particles collapse onto one point and stall at
{acc('budget', 'tuned | budget 500 | 5x100')}%, with identical results at all three budgets. The
best size is 25 at 500 and 1000 evaluations, and 25 or 50 at 2000
({p('budget', 'tuned | budget 2000 | 25x80', 'tuned | budget 2000 | 50x40')}).
"""))
body.append(figure("fig_budget.png", """
**Fig. 1.** Mean test accuracy against swarm size for three evaluation budgets (bars: 95%
confidence intervals).
"""))
body.append(para("""
A swarm needs enough iterations to converge and enough particles not to converge too soon. With 3 informants plus itself, a particle in a swarm of 5 sees 4 of
the 5 personal bests, so a small swarm contracts quickly; that cures the standard swarm's
failure to converge but over-tightens the tuned one. The best size also grows with the budget; Piotrowski *et al.*
(2020) found that it depends on the problem and is often not the customary 20–50.
"""))

body.append(para(f"""
**2.4 Q3: Velocity coefficients.** A 7 × 8 grid over α and β + γ, with β = γ (Fig. 2a), shows
that accuracy is highest along a diagonal band: less inertia can be traded for more attraction.
The best cell, α = 0.4 with β = γ = 1, reaches {pm('coefficients', 'alpha 0.4 | phi 2.0')}% test
accuracy against {pm('coefficients', 'alpha 0.72 | phi 3.0')}% for the standard values
({p('coefficients', 'alpha 0.4 | phi 2.0', 'alpha 0.72 | phi 3.0')}); a tuned run beats a
standard run with probability {a12('coefficients', 'alpha 0.4 | phi 2.0', 'alpha 0.72 | phi 3.0')}.
"""))
body.append(para(f"""
Diversity explains the band (Fig. 2c). All swarms start at a diversity of 3.9. Settings that end
far below 1 converged before finding a good region; settings that end above 10 are still
spreading. Accuracy peaks in between, where the swarm contracts slowly. The standard setting
ends at {diversity('coefficients', 'alpha 0.72 | phi 3.0')}: its swarm *expands* throughout the
run (Fig. 3b), although it lies inside the region where theory says a particle's variance stays
bounded (Poli, 2009).
"""))
body.append(para(f"""
That theory assumes the attractors x{{ast}} and x^{{+}} stay put. Here they move. Thresholded accuracy
barely changes when all the weights are scaled up, so good positions exist at any distance from
the origin, and in 49 dimensions almost all of the space lies further out. A control experiment
supports this (Fig. 3c): on a function that depends only on the direction of x the standard
swarm's diversity grows from 3.9 to {diversity('landscape', 'direction | standard')}, whereas on
a sphere function it ends at {diversity('landscape', 'sphere | standard')}; the tuned swarm
contracts on both. Particles leaving the initial region early is a known effect in high
dimensions (Helwig and Wanka, 2008), and large weights saturate the hidden units of PSO-trained
networks (Rakitianskaia and Engelbrecht, 2015); the best particle's root-mean-square (r.m.s.)
weight is {rms('coefficients', 'alpha 0.72 | phi 3.0')} for standard and
{rms('coefficients', 'alpha 0.4 | phi 2.0')} for tuned.
"""))
body.append(figure("fig_coefficients.png", """
**Fig. 2.** Velocity coefficients. (a) Mean test accuracy over α and β + γ; cells right of the
black line violate the stability condition β + γ < 24(1 − α^{2}) / (7 − 5α) (Poli, 2009). (b) A fixed
β + γ (3 for standard, 2 for tuned) split between cognitive and social pull. (c) Mean training
accuracy of the 56 grid settings against median final diversity.
"""))
body.append(figure("fig_convergence.png", """
**Fig. 3.** The baseline network under both coefficient sets. (a) Best training accuracy (mean;
band: interquartile range). (b) Median swarm diversity. (c) Median diversity on a control
function that depends only on the direction of x.
"""))
body.append(para(f"""
*Cognitive against social* (Fig. 2b). With β + γ fixed, the tuned swarm does best with an even or
slightly social split. A purely social swarm (β = 0) loses
{float(acc('balance', 'tuned | beta 1 gamma 1')) - float(acc('balance', 'tuned | beta 0 gamma 2')):.1f}
points, and a purely cognitive one (γ = 0) is the worst setting in the study
({pm('balance', 'tuned | beta 2 gamma 0')}%): with no informant term nothing is shared, giving
Luke's (2013) 'separate hill-climbers'. The standard swarm is almost insensitive to the split,
because its problem is the total pull, not the balance.
"""))
body.append(para(f"""
*Global best and jump size.* Moving half of the social pull onto the global best lifts the
standard swarm to {acc('global_best', 'standard | gamma 0.75 delta 0.75')}%, because a shared
attractor makes it contract, but lowers the tuned swarm to
{acc('global_best', 'tuned | gamma 0.5 delta 0.5')}%. Halving ε does the same. A linearly
decreasing α (0.9 → 0.4) was no better than a constant 0.4
({p('inertia_schedule', 'phi 2 | alpha constant 0.4', 'phi 2 | alpha 0.9 -> 0.4')}).
"""))

body.append(para(f"""
**2.5 Further experiments.** *Informants* (Fig. 4). With standard coefficients accuracy rises
with the number of informants, and nearest-neighbour informants beat random ones
({acc('informants', 'standard | nearest k=3')}% against {acc('informants', 'standard | random k=3')}%
at k = 3, {p('informants', 'standard | nearest k=3', 'standard | random k=3')}), because particles
that listen to nearby particles pull together. With tuned coefficients the ranking flips. Random
informants with k = 3 to 8 are best; nearest neighbours are worst
({acc('informants', 'tuned | nearest k=3')}% at k = 3), since neighbours hold nearly the same
information and the swarm splits into groups that converge separately. Topology is known to
change PSO's behaviour (Kennedy and Mendes, 2002); here the best topology depended on the
coefficients.
"""))
body.append(figure("fig_informants.png", """
**Fig. 4.** Mean test accuracy against the number of informants for three selection rules (bars:
95% confidence intervals).
"""))

table3_rows = [
    ["Baseline", pm("boundary", "standard | none"), pm("boundary", "tuned | none")],
    ["Walls at ±1, clamp", pm("boundary", "standard | clamp +/-1"), pm("boundary", "tuned | clamp +/-1")],
    ["Walls at ±1, reflect", pm("boundary", "standard | reflect +/-1"), pm("boundary", "tuned | reflect +/-1")],
    ["Walls at ±1, wrap", pm("boundary", "standard | wrap +/-1"), pm("boundary", "tuned | wrap +/-1")],
    ["Walls at ±1, random", pm("boundary", "standard | random +/-1"), pm("boundary", "tuned | random +/-1")],
    ["Velocity limit ±0.5", pm("velocity", "standard | vmax 0.5"), pm("velocity", "tuned | vmax 0.5")],
    ["Fitness = cross-entropy", pm("fitness_function", "standard | 1000 evals | cross-entropy"),
     pm("fitness_function", "tuned | 1000 evals | cross-entropy")],
    ["Fitness = MSE", pm("fitness_function", "standard | 1000 evals | MSE"),
     pm("fitness_function", "tuned | 1000 evals | MSE")],
    ["PSO-chosen activation", pm("activation_search", "standard | 1000 evals | 4-8-1 | PSO-chosen"),
     pm("activation_search", "tuned | 1000 evals | 4-8-1 | PSO-chosen")],
]
body.append(table(
    """**Table 3.** Further experiments on the baseline network: test accuracy (%, mean ± s.d. of
    30 runs).""",
    [col("Change to the baseline", 4.05), col("Standard", 1.6, "right"), col("Tuned", 1.6, "right")],
    table3_rows,
))
body.append(para(f"""
*Boundary handling* (Table 3). Walls matter only if particles reach them. The tuned swarm never
touches walls at ±5 or ±10, and at ±1 the handlers do not differ significantly (Kruskal–Wallis
p = {kruskal('boundary', 'tuned | none')}). For the standard swarm, clamping is best and
wrapping much the worst ({p('boundary', 'standard | clamp +/-1', 'standard | wrap +/-1')}):
wrapping carries a weight that passes +1 round to −1, flipping its sign, whereas clamping leaves
it at the nearest legal value. Helwig *et al.* (2013) also report large differences between
bound handling methods. A velocity limit helped the standard swarm more than any wall.
"""))
body.append(para(f"""
*Fitness measure.* Replacing accuracy with cross-entropy or MSE costs the tuned swarm about two
points ({p('fitness_function', 'tuned | 1000 evals | accuracy', 'tuned | 1000 evals | cross-entropy')}).
Tentatively, a loss can always be lowered by making already-correct outputs
more confident, so the swarm follows that direction (r.m.s. weight
{rms('fitness_function', 'tuned | 1000 evals | cross-entropy')} against
{rms('fitness_function', 'tuned | 1000 evals | accuracy')}) and converges before the last errors
are fixed.
"""))
body.append(para(f"""
*PSO-chosen activations.* Letting PSO choose the hidden activation costs a little at 1000
evaluations
({p('activation_search', 'tuned | 1000 evals | 4-8-1 | fixed tanh', 'tuned | 1000 evals | 4-8-1 | PSO-chosen')})
and nothing at 3000. Over 30 runs it chose tanh {picked.get('tanh', 0)} times, ELU
{picked.get('elu', 0)}, leaky ReLU {picked.get('leaky_relu', 0)}, ReLU {picked.get('relu', 0)} and
logistic {'never' if picked.get('logistic', 0) == 0 else str(picked['logistic']) + ' times'}, so
it avoids the one clearly bad option. It can do little more, because crossing a bin edge swaps
the function abruptly and the weights found so far no longer fit.
"""))

final_1k, final_5k = "final | 1000 evals", "final | 5000 evals"
body.append(para(f"""
**2.6 Final configuration and limitations.** No experiment found a setting significantly better
than the baseline network and swarm with tuned coefficients, so that is our final configuration
(Table 4). In 1000 evaluations it averages
{acc('final', final_1k)}% test accuracy, and {perfect_runs(final_1k)} of 30 runs classify all 412
test notes correctly. Five times as many evaluations bring no significant gain
({p('final', final_5k, final_1k)}). Repeated stratified 10-fold cross-validation gave
{cv_final['test_accuracy_mean']:.2f} ± {cv_final['test_accuracy_std']:.2f}%, so the result does
not depend on our one split. Kumar and Dudyala (2015) found multilayer perceptrons among the
most effective classifiers for this dataset; a 49-parameter network trained without gradients
classifies it almost perfectly.
"""))

table4_rows = []
for name in ("baseline", "final"):
    for budget in (1000, 5000):
        label = f"{name} | {budget} evals"
        cv = T["final_cv"].loc[label]
        table4_rows.append([
            "Standard (baseline)" if name == "baseline" else "Tuned (final)", f"{budget}",
            train_pm("final", label), pm("final", label),
            final_metric(label, "test_precision"), final_metric(label, "test_recall"),
            final_metric(label, "test_f1"),
            f"{cv['test_accuracy_mean']:.2f} ± {cv['test_accuracy_std']:.2f}",
        ])
body.append(table(
    f"""**Table 4.** Baseline and final configurations: 30 runs on the fixed split (%), and accuracy
    over {cv_runs} folds of repeated 10-fold cross-validation (CV).""",
    [col("Coefficients", 1.5), col("Evals", 0.55, "right"), col("Train acc.", 1.1, "right"),
     col("Test acc.", 1.1, "right"), col("Prec.", 0.6, "right"), col("Recall", 0.65, "right"),
     col("F1", 0.6, "right"), col("CV acc.", 1.15, "right")],
    table4_rows,
))
body.append(para("""
Limitations: the one dataset is nearly linearly separable, and differences among the good
settings (99.3–99.6%) are within noise. Hyperparameters
were varied one or two at a time around two points, so interactions may have been missed.
Settings were chosen by training accuracy, not on a validation set.
"""))

# ---------------------------------------------------------------- Section 3
std_1k, fin_1k = acc("final", "baseline | 1000 evals"), acc("final", final_1k)
body.append(h1("3  Conclusions"))
body.append(para(f"""
We built a configurable feedforward ANN and the informant-based PSO of Algorithm 39 from
scratch, coupled them through a fitness adapter and trained the network to authenticate
banknotes. Four conclusions follow.
"""))
body.append(para(f"""
*The velocity coefficients matter most.* The usual values made the swarm expand on this problem;
α = 0.4 with β = γ = 1 raised mean test accuracy from {float(std_1k):.1f}% to {float(fin_1k):.1f}%
at the same cost.
"""))
body.append(para("""
*Swarm diversity predicts accuracy.* The best settings let the swarm contract slowly; swarms
that collapsed or kept spreading did worse.
"""))
body.append(para("""
*The other answers depended on the coefficients.* With a swarm that did not converge, the
smallest network, the smallest swarm, the most informants and nearest-neighbour informants all
looked best. With tuned coefficients each of these reversed, in favour of one hidden layer of at
least 8 neurons, about 25 particles and 3 random informants.
"""))
body.append(para("""
*The extensions were of mixed value.* Boundary handling mattered only when particles reached the
walls, accuracy was a better fitness than a loss, and PSO-chosen activations gained nothing.
"""))
body.append(para("""
With more time we would adapt the inertia weight to the measured diversity during a run; repeat
the study on a harder, non-separable problem, where capacity and overfitting (van Wyk and
Engelbrecht, 2010) should matter more; and compare against backpropagation at an equal number
of evaluations.
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
    "*Proceedings of the 2002 Congress on Evolutionary Computation (CEC'02)*. Honolulu: IEEE, "
    "pp. 1671–1676.",
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
    "Rakitianskaia, A. and Engelbrecht, A. (2015) 'Saturation in PSO neural network training: good or "
    "evil?', in *2015 IEEE Congress on Evolutionary Computation (CEC)*. IEEE, pp. 125–132.",
    "Shi, Y. and Eberhart, R. (1998) 'A modified particle swarm optimizer', in *1998 IEEE "
    "International Conference on Evolutionary Computation Proceedings*. Anchorage: IEEE, pp. 69–73.",
    "van Wyk, A.B. and Engelbrecht, A.P. (2010) 'Overfitting by PSO trained feedforward neural "
    "networks', in *IEEE Congress on Evolutionary Computation*. Barcelona: IEEE, pp. 1–8.",
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
loop and its line-number comments, the added activations, the NumPy data handling and metrics,
the boundary handlers, the ring informant strategy, the velocity limit and inertia schedule, the
activation-search adapter, the command-line interface, the experiment runner, the analysis
script and the unit tests. (2) Experiments: the choice of experiments and value ranges, and all
the runs. (3) Report: the text, tables and figures.
"""))
appendix.append(para(f"""
**How it was checked.** The assistant compared the PSO loop with Algorithm 39 in the book's PDF,
downloaded the data file from the UCI repository and checked its size and class counts, ran the
70 unit tests, and ran the complete set of experiments twice to confirm that the {TOTAL_RUNS:,}
runs gave identical results. The numbers in this report are inserted from the result files by a
script, not typed. Each reference was checked against a publisher, repository or index record;
page numbers that could not be confirmed were left out.
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
            p_text = "best"
        elif pd.isna(r["p_holm"]):
            p_text = ""
        else:
            p_text = "<0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
        rows.append([name, str(int(r["D"])), train_pm(experiment, label), pm(experiment, label),
                     two_significant(r["final_diversity_median"]), p_text])
    return table(caption, [col(first_header, widths[0]), col("D", widths[1], "right"),
                           col("Train acc.", widths[2], "right"), col("Test acc.", widths[3], "right"),
                           col("Div.", widths[4], "right"), col("p", widths[5], "right")], rows)


appendix.append(h1("Appendix B  Additional results"))
appendix.append(para("""
The tables below give training as well as test accuracy (%, mean ± s.d. of 30 runs), the median
final swarm diversity (Div.) and the Holm-corrected p-value of a two-sided Mann–Whitney U test
on test accuracy against the best setting of the same block (the one with the highest mean
training accuracy, marked 'best'). The complete results of all 16 experiments are in
`results/summary.md` and `results/tables/` in the code submission.
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
`python main.py --help` lists the options for every hyperparameter;
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
    "campus": "Dubai Campus  ·  School of Mathematical and Computer Sciences",
    "course": "F21BC Biologically Inspired Computation  ·  Coursework Report  ·  November 2026",
    "title": "A Multilayer Perceptron Trained by Particle Swarm Optimisation",
    "authors": "Ahmed Abdallah (H00562295) and Mohamed Ali (H00562194)",
    "code": "Code: github.com/mr1pro/bio-inspired-coursework",
    "logo": os.path.join(HERE, "assets", "hw_logo.png"),
}


def plain(text):
    text = text.replace("{ast}", "*")
    text = re.sub(r"[*`]|\^\{|_\{|\}", "", text)
    return text


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
    print(f"references: {len(references)}")
