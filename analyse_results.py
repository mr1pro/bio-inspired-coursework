"""Turn the raw experiment results into tables, significance tests and figures.

Usage:
    python analyse_results.py

Reads   results/raw/*.csv and *_curves.npz   (written by run_experiments.py)
Writes  results/tables/<experiment>.csv      mean, standard deviation, tests
        results/summary.md                   every table in one readable file
        results/figures/*.png                the figures used in the report

Statistics. Accuracies are bounded at 100% and often skewed, so the tests are
rank-based (Derrac et al., 2011): within a block of comparable settings, each
setting is compared with the best one (highest mean training accuracy) using
a two-sided Mann-Whitney U test on test accuracy, with Holm's correction for
the number of comparisons in the block. A12 is the Vargha-Delaney effect size:
the probability that a run of the best setting beats a run of the other one.
"""

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "results", "raw")
TABLES = os.path.join(ROOT, "results", "tables")
FIGURES = os.path.join(ROOT, "results", "figures")

# Columns that define a block of settings that may be compared with each other.
BLOCKS: Dict[str, List[str]] = {
    "coefficients": [],
    "balance": ["pso"],
    "global_best": ["pso"],
    "jump_size": ["pso"],
    "architecture": ["pso"],
    "activation": ["pso", "architecture"],
    "budget": ["pso", "budget"],
    "informants": ["pso"],
    "boundary": ["pso"],
    "velocity": ["pso"],
    "inertia_schedule": ["phi"],
    "fitness_function": ["pso", "budget"],
    "activation_search": ["pso", "budget", "architecture"],
    "final": ["budget"],
    "final_cv": ["budget"],
    "landscape": ["function"],
}


# ----------------------------------------------------------------------
# Tables and statistics
# ----------------------------------------------------------------------
def load(name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(RAW, f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def load_curves(name: str) -> Dict[str, np.ndarray]:
    path = os.path.join(RAW, f"{name}_curves.npz")
    return dict(np.load(path)) if os.path.exists(path) else {}


def a12(better: Sequence[float], other: Sequence[float]) -> float:
    """Vargha-Delaney A: P(better > other) + 0.5 * P(equal)."""
    better = np.asarray(better)[:, None]
    other = np.asarray(other)[None, :]
    return float(np.mean(better > other) + 0.5 * np.mean(better == other))


def holm(p_values: Sequence[float]) -> List[float]:
    """Holm's step-down correction; returns adjusted p-values in input order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Mann-Whitney U p-value (1.0 if the samples are identical)."""
    if np.array_equal(np.sort(a), np.sort(b)):
        return 1.0
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def summarise(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """One row per setting: means, standard deviations and tests."""
    factor_columns = [c for c in df.columns[1:list(df.columns).index("seed")]]
    rows = []
    for label, group in df.groupby("label", sort=False):
        row = {"label": label}
        row.update({c: group[c].iloc[0] for c in factor_columns})
        row["runs"] = len(group)
        if "dimension" in group:
            row["D"] = int(group["dimension"].iloc[0])
        for metric in ("train_accuracy", "test_accuracy", "test_f1"):
            if metric in group:
                row[f"{metric}_mean"] = 100 * group[metric].mean()
                row[f"{metric}_std"] = 100 * group[metric].std(ddof=1)
        if "test_accuracy" in group:
            row["test_accuracy_min"] = 100 * group["test_accuracy"].min()
            row["test_accuracy_max"] = 100 * group["test_accuracy"].max()
        if "best_fitness" in group and "test_accuracy" not in group:
            row["best_fitness_mean"] = group["best_fitness"].mean()
            row["best_fitness_std"] = group["best_fitness"].std(ddof=1)
        for metric in ("final_diversity", "weight_norm"):
            if metric in group:
                row[f"{metric}_median"] = group[metric].median()
        rows.append(row)
    table = pd.DataFrame(rows)

    if "test_accuracy" not in df:
        return table

    # Tests against the best setting of each block.
    table["best_in_block"] = False
    table["p_holm"] = np.nan
    table["A12"] = np.nan
    table["kruskal_p"] = np.nan
    block_columns = BLOCKS.get(name, [])
    blocks = table.groupby(block_columns, sort=False) if block_columns else [(None, table)]
    for _, block in blocks:
        if len(block) < 2:
            continue
        samples = {label: df.loc[df["label"] == label, "test_accuracy"].to_numpy()
                   for label in block["label"]}
        ranked = block.sort_values(["train_accuracy_mean", "test_accuracy_mean"], ascending=False)
        best = ranked["label"].iloc[0]
        others = [label for label in block["label"] if label != best]
        raw_p = [mann_whitney(samples[best], samples[label]) for label in others]
        adjusted = holm(raw_p)
        try:
            kruskal = float(stats.kruskal(*samples.values()).pvalue)
        except ValueError:
            kruskal = 1.0
        table.loc[table["label"] == best, "best_in_block"] = True
        table.loc[table["label"].isin(block["label"]), "kruskal_p"] = kruskal
        for label, p in zip(others, adjusted):
            table.loc[table["label"] == label, "p_holm"] = p
            table.loc[table["label"] == label, "A12"] = a12(samples[best], samples[label])
    return table


def compare(df: pd.DataFrame, label_a: str, label_b: str, metric: str = "test_accuracy") -> Dict:
    """Direct comparison of two settings (used for statements in the report)."""
    a = df.loc[df["label"] == label_a, metric].to_numpy()
    b = df.loc[df["label"] == label_b, metric].to_numpy()
    return {"a": label_a, "b": label_b, "metric": metric,
            "mean_a": 100 * a.mean(), "mean_b": 100 * b.mean(),
            "p": mann_whitney(a, b), "A12": a12(a, b)}


def to_markdown(table: pd.DataFrame) -> str:
    """Compact Markdown table: mean +/- std columns and the test results."""
    lines = ["| setting | runs | D | train acc. % | test acc. % | test F1 % | diversity | weight RMS | p (Holm) | A12 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        def pm(metric):
            if f"{metric}_mean" not in r or pd.isna(r[f"{metric}_mean"]):
                return ""
            return f"{r[f'{metric}_mean']:.2f} ± {r[f'{metric}_std']:.2f}"
        if r.get("best_in_block", False):
            p_text = "best"
        elif pd.isna(r.get("p_holm", np.nan)):
            p_text = ""
        else:
            p_text = "<0.001" if r["p_holm"] < 0.001 else f"{r['p_holm']:.3f}"
        a_text = "" if pd.isna(r.get("A12", np.nan)) else f"{r['A12']:.2f}"
        d_text = "" if pd.isna(r.get("D", np.nan)) else f"{int(r['D'])}"
        div = r.get("final_diversity_median", np.nan)
        rms = r.get("weight_norm_median", np.nan)
        lines.append(
            f"| {r['label']} | {int(r['runs'])} | {d_text} | {pm('train_accuracy')} | {pm('test_accuracy')} "
            f"| {pm('test_f1')} | {'' if pd.isna(div) else f'{div:.3g}'} | {'' if pd.isna(rms) else f'{rms:.3g}'} "
            f"| {p_text} | {a_text} |"
        )
    return "\n".join(lines)


# Pairs of settings whose difference is discussed in the report. Each is
# tested directly (two-sided Mann-Whitney U on test accuracy, no correction).
KEY_COMPARISONS = [
    ("final", "final | 1000 evals", "baseline | 1000 evals"),
    ("final", "final | 5000 evals", "baseline | 5000 evals"),
    ("final", "final | 5000 evals", "final | 1000 evals"),
    ("final_cv", "final | 1000 evals", "baseline | 1000 evals"),
    ("coefficients", "alpha 0.4 | phi 2.0", "alpha 0.72 | phi 3.0"),
    ("balance", "tuned | beta 1 gamma 1", "tuned | beta 0 gamma 2"),
    ("balance", "tuned | beta 1 gamma 1", "tuned | beta 2 gamma 0"),
    ("balance", "standard | beta 1.5 gamma 1.5", "standard | beta 0 gamma 3"),
    ("balance", "standard | beta 1.5 gamma 1.5", "standard | beta 3 gamma 0"),
    ("global_best", "standard | gamma 0.75 delta 0.75", "standard | gamma 1.5 delta 0"),
    ("global_best", "tuned | gamma 1 delta 0", "tuned | gamma 0.5 delta 0.5"),
    ("jump_size", "standard | epsilon 0.5", "standard | epsilon 1.0"),
    ("jump_size", "tuned | epsilon 1.0", "tuned | epsilon 0.5"),
    ("architecture", "standard | 4-1", "standard | 4-8-1"),
    ("architecture", "standard | 4-1", "standard | 4-8-8-8-1"),
    ("architecture", "tuned | 4-8-1", "tuned | 4-1"),
    ("architecture", "tuned | 4-8-1", "tuned | 4-8-8-1"),
    ("architecture", "tuned | 4-16-1", "tuned | 4-8-1"),
    ("activation", "tuned | 4-8-1 | relu", "tuned | 4-8-1 | tanh"),
    ("activation", "tuned | 4-8-1 | tanh", "tuned | 4-8-1 | logistic"),
    ("activation", "tuned | 4-8-1 | tanh", "tuned | 4-8-1 | linear"),
    ("activation", "tuned | 4-8-8-1 | tanh", "tuned | 4-8-8-1 | logistic"),
    ("activation", "standard | 4-8-1 | linear", "standard | 4-8-1 | tanh"),
    ("budget", "standard | budget 500 | 5x100", "standard | budget 500 | 25x20"),
    ("budget", "standard | budget 500 | 5x100", "standard | budget 500 | 250x2"),
    ("budget", "tuned | budget 500 | 25x20", "tuned | budget 500 | 5x100"),
    ("budget", "tuned | budget 500 | 25x20", "tuned | budget 500 | 10x50"),
    ("budget", "tuned | budget 1000 | 25x40", "tuned | budget 1000 | 50x20"),
    ("budget", "tuned | budget 2000 | 25x80", "tuned | budget 2000 | 50x40"),
    ("informants", "standard | nearest k=3", "standard | random k=3"),
    ("informants", "standard | random k=24", "standard | random k=3"),
    ("informants", "tuned | random k=3", "tuned | nearest k=3"),
    ("informants", "tuned | random k=3", "tuned | ring k=3"),
    ("informants", "tuned | random k=3", "tuned | random k=24"),
    ("boundary", "standard | clamp +/-1", "standard | none"),
    ("boundary", "standard | clamp +/-1", "standard | reflect +/-1"),
    ("boundary", "standard | clamp +/-1", "standard | wrap +/-1"),
    ("boundary", "standard | none", "standard | wrap +/-1"),
    ("velocity", "standard | vmax 0.5", "standard | vmax None"),
    ("velocity", "tuned | vmax None", "tuned | vmax 0.1"),
    ("velocity", "standard | zero initial velocity", "standard | vmax None"),
    ("inertia_schedule", "phi 2 | alpha constant 0.4", "phi 2 | alpha 0.9 -> 0.4"),
    ("fitness_function", "tuned | 1000 evals | accuracy", "tuned | 1000 evals | cross-entropy"),
    ("fitness_function", "tuned | 1000 evals | accuracy", "tuned | 1000 evals | MSE"),
    ("fitness_function", "standard | 1000 evals | accuracy", "standard | 1000 evals | cross-entropy"),
    ("activation_search", "tuned | 1000 evals | 4-8-1 | fixed tanh", "tuned | 1000 evals | 4-8-1 | PSO-chosen"),
    ("activation_search", "tuned | 3000 evals | 4-8-1 | fixed tanh", "tuned | 3000 evals | 4-8-1 | PSO-chosen"),
]


def key_comparisons(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run every comparison in KEY_COMPARISONS for which data is available."""
    rows = []
    for experiment, label_a, label_b in KEY_COMPARISONS:
        df = data.get(experiment)
        if df is None or not {label_a, label_b} <= set(df["label"]):
            continue
        result = compare(df, label_a, label_b)
        rows.append({"experiment": experiment, "A": label_a, "B": label_b,
                     "test_acc_A": round(result["mean_a"], 2), "test_acc_B": round(result["mean_b"], 2),
                     "p_value": result["p"], "A12": round(result["A12"], 3)})
    return pd.DataFrame(rows)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Plain Markdown table for any small data frame."""
    def cell(value):
        return f"{value:.4g}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
# Colours: fixed categorical order (blue, orange, aqua), an ordered blue ramp
# for ordered quantities, and a one-hue ramp for the heatmap. Marker shapes
# repeat the colour coding so the plots also work in greyscale.
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
RAMP3 = ["#84b9f3", "#257de0", "#0f4a96"]
INK, INK_SOFT, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cce3fd", "#84b9f3", "#338ced", "#1757ac", "#0e3d7e"])
PSO_STYLE = {"standard": (ORANGE, "s", "standard coefficients"),
             "tuned": (BLUE, "o", "tuned coefficients")}
# Expected distance from the centre of 25 points drawn uniformly from
# [-1, 1]^49, i.e. the diversity of the swarm before it has moved.
INITIAL_DIVERSITY = 3.95


def setup_style() -> None:
    family = "Carlito" if any("Carlito" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 2.0, "lines.markersize": 6.5,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


def ci95(values: np.ndarray) -> float:
    """Half-width of a normal-approximation 95% confidence interval of the mean."""
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def save(fig, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, f"{name}.png"))
    plt.close(fig)


def stable_limit(alpha: float) -> float:
    """Largest phi = beta + gamma for which the particle variance stays
    bounded while the swarm stagnates (Poli, 2009):
    phi < 24 (1 - alpha^2) / (7 - 5 alpha)."""
    return 24.0 * (1.0 - alpha ** 2) / (7.0 - 5.0 * alpha)


def figure_coefficients(df: pd.DataFrame, balance: pd.DataFrame) -> None:
    """(a) test accuracy over the alpha/phi grid, (b) cognitive against social
    pull, (c) training accuracy of the 56 grid settings against how spread
    out the swarm was at the end of the run."""
    grouped = df.groupby(["alpha", "phi"])
    test = (100 * grouped["test_accuracy"].mean()).unstack()
    train = 100 * grouped["train_accuracy"].mean()
    diversity = grouped["final_diversity"].median()
    alphas, phis = list(test.index), list(test.columns)

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(7.4, 2.95),
                                       gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # (a) heatmap, one hue, darker = more accurate
    ax.imshow(test.to_numpy(), cmap=SEQUENTIAL, aspect="auto", origin="lower", vmin=93, vmax=100)
    for i in range(len(alphas)):
        for j in range(len(phis)):
            value = test.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.0,
                    color="white" if value > 97.6 else INK)
    # Step line: cells to the right of it violate the stability condition.
    xs, ys = [], []
    for i, alpha in enumerate(alphas):
        unstable = [j for j, phi in enumerate(phis) if phi >= stable_limit(alpha)]
        edge = (unstable[0] - 0.5) if unstable else len(phis) - 0.5
        xs += [edge, edge]
        ys += [i - 0.5, i + 0.5]
    ax.plot(xs, ys, color=INK, linewidth=1.6, solid_joinstyle="miter", clip_on=False)
    ax.set_xticks(range(len(phis)), [f"{p:g}" for p in phis])
    ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlabel("β + γ  (with β = γ)")
    ax.set_ylabel("α (velocity retained)")
    ax.set_title("(a) Test accuracy (%)", loc="left", pad=12)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # (b) cognitive share
    for pso in ("standard", "tuned"):
        colour, marker, _ = PSO_STYLE[pso]
        part = balance[balance["pso"] == pso]
        shares = sorted(part["cognitive_share"].unique())
        means = [100 * part.loc[part["cognitive_share"] == s, "test_accuracy"].mean() for s in shares]
        errors = [100 * ci95(part.loc[part["cognitive_share"] == s, "test_accuracy"].to_numpy()) for s in shares]
        ax1.errorbar(shares, means, yerr=errors, color=colour, marker=marker, markeredgecolor="white",
                     markeredgewidth=0.8, capsize=2.5, elinewidth=1.0, label=pso)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "", "0.5", "", "1"])
    ax1.set_xlabel("Cognitive share  β / (β + γ)")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_ylim(84, 100.4)
    ax1.set_yticks([85, 90, 95, 100])
    ax1.set_title("(b) Cognitive vs. social", loc="left", pad=12)
    ax1.legend(loc="lower left", handlelength=1.4, borderaxespad=0.3)

    # (c) accuracy against swarm spread: every grid setting in grey, the two
    # parameter sets used in the rest of the study picked out
    ax2.scatter(diversity.to_numpy(), train.to_numpy(), s=24, color="#9c9c9c", edgecolor="white",
                linewidth=0.7, zorder=3, label="grid setting")
    for pso, key in (("standard", (0.72, 3.0)), ("tuned", (0.4, 2.0))):
        colour, marker, _ = PSO_STYLE[pso]
        ax2.scatter([diversity[key]], [train[key]], s=62, color=colour, marker=marker,
                    edgecolor="white", linewidth=1.0, zorder=4, label=pso)
    ax2.axvline(INITIAL_DIVERSITY, color=AXIS, linewidth=1.0, zorder=1)
    ax2.text(INITIAL_DIVERSITY, 100.25, "start", fontsize=8.5, color=INK_SOFT, ha="center", va="bottom")
    ax2.set_xscale("log")
    ax2.set_xticks([0.1, 10, 1000])
    ax2.minorticks_off()
    ax2.set_xlabel("Final swarm diversity")
    ax2.set_ylabel("Training accuracy (%)")
    ax2.set_title("(c) Accuracy vs. spread", loc="left", pad=12)
    ax2.set_ylim(93, 100.2)
    ax2.legend(loc="lower right", handlelength=1.0, borderaxespad=0.2, handletextpad=0.4)
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_coefficients")


def figure_budget(df: pd.DataFrame) -> None:
    """Test accuracy against swarm size for three evaluation budgets."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    sizes = sorted(df["swarm_size"].unique())
    markers = ["^", "o", "s"]
    for ax, pso in zip(axes, ("standard", "tuned")):
        for colour, marker, budget in zip(RAMP3, markers, sorted(df["budget"].unique())):
            part = df[(df["pso"] == pso) & (df["budget"] == budget)]
            means = [100 * part.loc[part["swarm_size"] == n, "test_accuracy"].mean() for n in sizes]
            errors = [100 * ci95(part.loc[part["swarm_size"] == n, "test_accuracy"].to_numpy()) for n in sizes]
            ax.errorbar(sizes, means, yerr=errors, color=colour, marker=marker, markeredgecolor="white",
                        markeredgewidth=0.8, capsize=2.5, elinewidth=1.0, label=f"{budget} evaluations")
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Swarm size (iterations = budget / swarm size)")
        ax.set_title(f"({'ab'[list(axes).index(ax)]}) {PSO_STYLE[pso][2].capitalize()}", loc="left")
    axes[0].set_ylabel("Mean test accuracy (%)")
    axes[0].set_ylim(88, 100.3)
    axes[1].legend(loc="lower right", handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_budget")


def figure_convergence(curves: Dict[str, np.ndarray], landscape_curves: Dict[str, np.ndarray]) -> None:
    """Training accuracy and swarm diversity over a run, baseline vs tuned,
    plus the diversity of the same two swarms on the scale-free test function."""
    labels = {"standard": "standard | 4-8-1", "tuned": "tuned | 4-8-1"}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for pso, label in labels.items():
        colour, marker, name = PSO_STYLE[pso]
        fitness = 100 * curves[f"{label}::fitness_history"]
        diversity = curves[f"{label}::diversity_history"]
        iterations = np.arange(1, fitness.shape[1] + 1)
        axes[0].plot(iterations, fitness.mean(axis=0), color=colour, label=name)
        axes[0].fill_between(iterations, np.percentile(fitness, 25, axis=0),
                             np.percentile(fitness, 75, axis=0), color=colour, alpha=0.12, linewidth=0)
        axes[1].plot(iterations, np.median(diversity, axis=0), color=colour, label=name)
        control = landscape_curves.get(f"direction | {pso}::diversity_history")
        if control is not None:
            axes[2].plot(np.arange(1, control.shape[1] + 1), np.median(control, axis=0), color=colour, label=name)
    axes[0].set_ylim(90, 100.2)
    axes[0].set_ylabel("Best training accuracy (%)")
    axes[0].set_title("(a) Convergence, 4-8-1 network", loc="left")
    axes[0].legend(loc="lower right", handlelength=1.4)
    for ax, title in ((axes[1], "(b) Swarm diversity, ANN"), (axes[2], "(c) Diversity, control function")):
        ax.set_yscale("log")
        ax.set_ylabel("Mean distance from centre")
        ax.set_title(title, loc="left")
        ax.minorticks_off()
    for ax in axes:
        ax.set_xlabel("Iteration")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_convergence")


def figure_informants(df: pd.DataFrame) -> None:
    """Test accuracy against number of informants for three selection rules."""
    styles = {"random": (BLUE, "o", "random (re-drawn)"), "nearest": (ORANGE, "s", "nearest neighbours"),
              "ring": (AQUA, "^", "ring")}
    counts = sorted(df["informants"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    for index, (ax, pso) in enumerate(zip(axes, ("standard", "tuned"))):
        for strategy, (colour, marker, name) in styles.items():
            part = df[(df["pso"] == pso) & (df["strategy"] == strategy)]
            means = [100 * part.loc[part["informants"] == k, "test_accuracy"].mean() for k in counts]
            errors = [100 * ci95(part.loc[part["informants"] == k, "test_accuracy"].to_numpy()) for k in counts]
            ax.errorbar(counts, means, yerr=errors, color=colour, marker=marker, markeredgecolor="white",
                        markeredgewidth=0.8, capsize=2.5, elinewidth=1.0, label=name)
        ax.set_xscale("log")
        ax.set_xticks(counts, [str(k) for k in counts])
        ax.minorticks_off()
        ax.set_xlabel("Informants per particle (swarm of 25)")
        ax.set_title(f"({'ab'[index]}) {PSO_STYLE[pso][2].capitalize()}", loc="left")
    axes[0].set_ylabel("Mean test accuracy (%)")
    axes[0].legend(loc="lower right", handlelength=1.6)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_informants")


# ----------------------------------------------------------------------
def main() -> None:
    os.makedirs(TABLES, exist_ok=True)
    setup_style()
    sections = ["# Experiment results\n",
                "All accuracies in percent, mean ± sample standard deviation over the runs. "
                "`p (Holm)` is a two-sided Mann-Whitney U test on test accuracy against the best "
                "setting of the same block (the one with the highest mean training accuracy), "
                "Holm-corrected within the block. `A12` is the probability that a run of the best "
                "setting beats a run of this one. `diversity` is the median distance of the particles "
                "from the swarm centre at the end of the run; `weight RMS` is the median root-mean-square "
                "of the best particle's coordinates.\n"]
    data: Dict[str, pd.DataFrame] = {}
    for name in BLOCKS:
        df = load(name)
        if df is None:
            continue
        data[name] = df
        table = summarise(name, df)
        table.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False, float_format="%.4f")
        sections.append(f"## {name}\n")
        if "test_accuracy_mean" in table:
            if "kruskal_p" in table and table["kruskal_p"].notna().any():
                p_values = table["kruskal_p"].dropna().unique()
                sections.append("Kruskal-Wallis p per block: "
                                + ", ".join("<0.001" if p < 0.001 else f"{p:.3f}" for p in p_values) + "\n")
            sections.append(to_markdown(table) + "\n")
        else:
            sections.append(frame_to_markdown(table) + "\n")
        print(f"table: {name} ({len(table)} settings)")

    if "activation_search" in data:
        df = data["activation_search"]
        chosen = df[df["activations"] == "PSO-chosen"]
        counts = (chosen.groupby(["pso", "budget", "architecture"])["chosen_activations"]
                  .value_counts().rename("runs").reset_index())
        counts.to_csv(os.path.join(TABLES, "activation_search_choices.csv"), index=False)
        sections.append("## activation_search: activations chosen by PSO\n")
        sections.append(frame_to_markdown(counts) + "\n")

    comparisons = key_comparisons(data)
    if len(comparisons):
        comparisons.to_csv(os.path.join(TABLES, "key_comparisons.csv"), index=False)
        sections.append("## Direct comparisons quoted in the report\n")
        sections.append("Two-sided Mann-Whitney U on test accuracy (uncorrected); "
                        "A12 = probability that a run of A beats a run of B.\n")
        sections.append(frame_to_markdown(comparisons) + "\n")

    with open(os.path.join(ROOT, "results", "summary.md"), "w") as handle:
        handle.write("\n".join(sections))

    if "coefficients" in data and "balance" in data:
        figure_coefficients(data["coefficients"], data["balance"])
    if "budget" in data:
        figure_budget(data["budget"])
    if "architecture" in data:
        figure_convergence(load_curves("architecture"), load_curves("landscape"))
    if "informants" in data:
        figure_informants(data["informants"])
    print("figures written to results/figures/")


if __name__ == "__main__":
    main()
