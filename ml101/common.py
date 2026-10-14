"""Shared helpers for the ML 101 demos: data loading, feature scaling and the
training animation used by all three models.

Notation follows the CS229 notes (main_notes.pdf, chapters 1 and 2):
    x^(i)    input features of example i, with x_0 = 1 (the intercept term)
    y^(i)    target, 0 or 1 for our binary Iris problem
    theta    parameters, one per feature (theta_0 is the intercept)
    h(x)     the hypothesis, i.e. the model's output
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["animation.embed_limit"] = 60  # MB; inline players embed every frame

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"

ALL_FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
DEFAULT_FEATURES = ("sepal_length", "sepal_width")
DEFAULT_CLASSES = ("setosa", "versicolor")  # class 0, class 1

CLASS_COLORS = ("#4656c9", "#d0861b")  # setosa-ish indigo, versicolor-ish ochre
CLASS_MARKERS = ("o", "^")

# A deliberately poor starting line, so the animation shows real movement.
# (In standardized units: intercept, weight for feature 1, weight for feature 2.)
INITIAL_THETA = np.array([0.0, 0.2, 1.0])


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    X_raw: np.ndarray        # (m, 2) features in centimetres
    y: np.ndarray            # (m,) labels in {0, 1}
    features: tuple[str, str]
    classes: tuple[str, str]
    mean: np.ndarray = field(init=False)
    std: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mean = self.X_raw.mean(axis=0)
        self.std = self.X_raw.std(axis=0)

    @property
    def X(self) -> np.ndarray:
        """Standardized features with the intercept column x_0 = 1 prepended."""
        return add_intercept(self.standardize(self.X_raw))

    def standardize(self, X_raw: np.ndarray) -> np.ndarray:
        return (X_raw - self.mean) / self.std


def load_binary_iris(features=DEFAULT_FEATURES, classes=DEFAULT_CLASSES) -> Dataset:
    """Load two Iris species and two features from data/iris.csv.

    The first species in `classes` gets label 0, the second gets label 1.
    """
    with open(DATA_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["species"] in classes]
    X_raw = np.array([[float(r[feat]) for feat in features] for r in rows])
    y = np.array([classes.index(r["species"]) for r in rows], dtype=float)
    return Dataset(X_raw, y, tuple(features), tuple(classes))


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def accuracy(predictions: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(predictions == y))


# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    step: int                    # iteration / update number
    theta: np.ndarray            # parameters at this point in training
    cost: float                  # value plotted on the right-hand panel
    highlight: int | None = None # index of the example that triggered an update


def pick_frames(n_steps: int, max_frames: int = 90) -> np.ndarray:
    """Choose which training steps to draw.

    Gradient descent moves fast at the start and slowly later, so the frames
    are spaced geometrically: dense early, sparse late.
    """
    if n_steps <= max_frames:
        return np.arange(n_steps)
    idx = np.unique(np.geomspace(1, n_steps, max_frames).astype(int) - 1)
    return np.concatenate([[0], idx[idx > 0]])


def animate_training(
    data: Dataset,
    history: list[Frame],
    predict_output,              # function(theta, X_with_intercept) -> h values in [0, 1]
    threshold: float,            # boundary is where theta^T x == threshold
    title: str,
    cost_label: str,
    frame_ids=None,
    extra_line: np.ndarray | None = None,
    extra_label: str = "",
    fps: int = 20,
    save: str | None = None,
):
    """Animate the decision boundary converging next to the cost curve.

    Returns an HTML player (play / pause / step / scrub) that Jupyter shows
    inline when it is the last expression of a cell. Pass save="file.gif"
    to also write a GIF for slides.
    """
    if frame_ids is None:
        frame_ids = pick_frames(len(history))

    X_raw, y = data.X_raw, data.y
    pad = 0.4
    x_min, x_max = X_raw[:, 0].min() - pad, X_raw[:, 0].max() + pad
    y_min, y_max = X_raw[:, 1].min() - pad, X_raw[:, 1].max() + pad

    # Grid in centimetres for the background shading of h(x).
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid_raw = np.column_stack([gx.ravel(), gy.ravel()])
    grid = add_intercept(data.standardize(grid_raw))

    cmap = LinearSegmentedColormap.from_list("iris", [CLASS_COLORS[0], "#f4f4f6", CLASS_COLORS[1]])

    fig, (ax, ax_cost) = plt.subplots(1, 2, figsize=(12, 5.8), dpi=72,
                                      gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold")

    shade = ax.imshow(np.zeros_like(gx), extent=(x_min, x_max, y_min, y_max), origin="lower",
                      cmap=cmap, vmin=0, vmax=1, alpha=0.35, aspect="auto", interpolation="bilinear")
    for label in (0, 1):
        pts = X_raw[y == label]
        ax.scatter(pts[:, 0], pts[:, 1], c=CLASS_COLORS[label], marker=CLASS_MARKERS[label],
                   edgecolor="white", linewidth=0.6, s=46, label=f"{data.classes[label]} (y={label})",
                   zorder=3)
    wrong = ax.scatter([], [], s=150, facecolor="none", edgecolor="#c0233b", linewidth=1.6,
                       label="misclassified", zorder=4)
    hl = ax.scatter([], [], s=320, facecolor="none", edgecolor="black", linewidth=2.2, zorder=5)
    (line,) = ax.plot([], [], color="black", linewidth=2.4, label="decision boundary", zorder=4)
    if extra_line is not None:
        xs, ys = boundary_segment(extra_line, threshold, data, (x_min, x_max), (y_min, y_max))
        ax.plot(xs, ys, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                label=extra_label, zorder=3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(pretty(data.features[0]))
    ax.set_ylabel(pretty(data.features[1]))
    # Legend below the scatter so it never hides a data point.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.5, frameon=False)

    steps = np.array([f.step for f in history])
    costs = np.array([f.cost for f in history])
    ax_cost.plot(steps, costs, color="#b5b8c4", linewidth=1.5)
    (trace,) = ax_cost.plot([], [], color="black", linewidth=2)
    (dot,) = ax_cost.plot([], [], "o", color="#c0233b")
    ax_cost.set_xlabel("iteration" if history[-1].highlight is None else "update")
    ax_cost.set_ylabel(cost_label)
    ax_cost.set_title("Cost during training", fontsize=11)
    ax_cost.grid(alpha=0.3)
    info = ax_cost.text(0.97, 0.96, "", transform=ax_cost.transAxes, fontsize=9, family="monospace",
                        ha="right", va="top", multialignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.tight_layout()

    def draw(k):
        f = history[frame_ids[k]]
        h_grid = predict_output(f.theta, grid).reshape(gx.shape)
        shade.set_data(np.clip(h_grid, 0, 1))
        line.set_data(*boundary_segment(f.theta, threshold, data, (x_min, x_max), (y_min, y_max)))

        pred = (predict_output(f.theta, data.X) >= 0.5).astype(float)
        wrong.set_offsets(X_raw[pred != y] if np.any(pred != y) else np.empty((0, 2)))
        hl.set_offsets(X_raw[[f.highlight]] if f.highlight is not None else np.empty((0, 2)))

        upto = steps <= f.step
        trace.set_data(steps[upto], costs[upto])
        dot.set_data([f.step], [f.cost])
        th = ", ".join(f"{t:+.2f}" for t in f.theta)
        info.set_text(f"step     {f.step:>5d}\ntheta    [{th}]\ncost     {f.cost:.4g}\n"
                      f"accuracy {accuracy(pred, y):.0%}")
        return shade, line, wrong, hl, trace, dot, info

    anim = FuncAnimation(fig, draw, frames=len(frame_ids), interval=1000 / fps, blit=False)
    if save:
        anim.save(save, writer=PillowWriter(fps=fps))
    player = HTML(anim.to_jshtml(default_mode="once"))
    plt.close(fig)  # keep Jupyter from also showing the last frame as a static image
    return player


def boundary_segment(theta, threshold, data: Dataset, xlim, ylim):
    """Endpoints of the line theta^T x = threshold, drawn in centimetres.

    theta lives in standardized units: theta_0 + theta_1*z_1 + theta_2*z_2,
    with z_j = (x_j - mean_j) / std_j. We convert the line back to cm.
    """
    t0, t1, t2 = theta
    (m1, m2), (s1, s2) = data.mean, data.std
    a, b = t1 / s1, t2 / s2                     # coefficients on x1, x2 in cm
    c = threshold - t0 + t1 * m1 / s1 + t2 * m2 / s2
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return [], []
    if abs(b) >= abs(a):                        # mostly horizontal: solve for x2
        xs = np.array(xlim)
        return xs, (c - a * xs) / b
    ys = np.array(ylim)                         # mostly vertical: solve for x1
    return (c - b * ys) / a, ys


def pretty(feature: str) -> str:
    return feature.replace("_", " ") + " (cm)"


def print_summary(name: str, theta: np.ndarray, data: Dataset, predictions: np.ndarray):
    print(f"\n{name}")
    print(f"  species : {data.classes[0]} (y=0) vs {data.classes[1]} (y=1)")
    print(f"  features: {data.features[0]}, {data.features[1]} (standardized)")
    print(f"  theta   : {np.array2string(theta, precision=3)}")
    print(f"  training accuracy: {accuracy(predictions, data.y):.1%}")
