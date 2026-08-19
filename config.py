"""
Configuration parameters for the optimal stopping model.
Ensures parameters are valid for the Bayesian sequential voting model.
"""

import numpy as np
from typing import TypedDict, List

class BoundShape(TypedDict):
    """
    One arrival-intensity shape compared in ``plot_bound_over_time``.

    Each shape has its own log-linear intensity λ(x) = exp(a + b·x). These are
    calibrated independently of the model's ``arrival_a``/``arrival_b`` so that
    several shapes can be overlaid at a common total expected arrivals Λ(T).

    Fields
    ------
    label : str
        Legend label for the comparison curve.
    a : float
        Intercept of the log-linear intensity (calibrated; NOT the model's arrival_a).
    b : float
        Shape parameter: b<0 front-loaded, b=0 uniform, b>0 back-loaded.
    ls : str
        Matplotlib linestyle for this curve (e.g. "-", ":", "--").
    """
    label: str
    a: float
    b: float
    ls: str

class Config(TypedDict):
    """
    Top-level configuration for a single model run.

    Fields group into core model parameters (N, T, c, R and the arrival
    intensity a, b), the boundary-shape comparison set, and plotting options.
    See ``validate_config`` for the invariants these values must satisfy.
    """
    num_voters: int      # N: Total population
    time_horizon: int    # T: Terminal date
    step_cost: float     # c: Sampling cost
    reward: float        # R: Reward for correct decision
    arrival_a: float     # a: intercept of log-linear intensity λ(x) = exp(a + b·x), x ∈ [0,1]
    arrival_b: float     # b: shape (b<0 front-loaded, b=0 uniform, b>0 back-loaded)
    bound_shapes: List[BoundShape]  # Arrival shapes compared in plot_bound_over_time (independent of arrival_a/b)
    plot_mode: str
    asm_track: str
    times_to_plot: List[int]

CONFIG: Config = {
    # Core model parameters (Symbols from LaTeX: N, T, c, R, a, b)
    "num_voters": 497,
    "time_horizon": 28,
    "step_cost": 0.001, #0.001,
    "reward": 100.0,
    "arrival_a": 1.7367, #1535330,                  # exp(0) = 1, so this reproduces λ_t ≡ 1 when b = 0
    "arrival_b": 0.0,                  # b=0 uniform; b<0 front-loaded; b>0 back-loaded

    # Boundary shape comparison (plot_bound_over_time): all calibrated to the same
    # total expected arrivals; independent of the model's arrival_a/arrival_b above.
    "bound_shapes": [
        {"label": "Front-loaded", "a": 2.9553, "b": -3.05, "ls": ":"},
        {"label": "Uniform",      "a": 1.7367, "b":  0.0, "ls": "-"},
        {"label": "Back-loaded",  "a": -0.2036, "b":  3.05, "ls": "--"},
    ],
    #{"label": "Front-loaded", "a": 2.011915, "b": -1.0, "ls": ":"},
    #    {"label": "Uniform",      "a": 1.535330, "b":  0.0, "ls": "-"},
    #    {"label": "Back-loaded",  "a": 0.976201, "b":  1.0, "ls": "--"},

    # Plotting & Analysis
    "plot_mode": "Diff-vs-Sum",        # Options: "Aye-vs-Nay", "Diff-vs-Sum", "Acceptance-vs-Support", "Acceptance-vs-Turnout"
    "asm_track": "",                   # Options: "", "default", "root", "big", "medium", "small"
    "times_to_plot": [2, 14, 26],
}

def validate_config(cfg: Config):
    """
    Check the model assumptions a config must satisfy before a run.

    Verifies that N is odd (unique majority), T is positive, every requested
    plot time lies within the horizon, and that total expected arrivals Λ(T)
    stay below N — both for the model's own (a, b) and for every comparison
    shape in ``bound_shapes``.

    Parameters
    ----------
    cfg : Config
        The configuration dictionary to validate.

    Raises
    ------
    AssertionError
        If any invariant is violated; the message names the offending value.
    """
    assert cfg["num_voters"] % 2 != 0, "N must be odd for a unique majority."
    assert cfg["time_horizon"] > 0, "T must be at least 1."
    assert all(t < cfg["time_horizon"] for t in cfg["times_to_plot"]), "Plot times must be within T."
    # Total expected arrivals over the horizon must be feasible (Λ(T) < N).
    T = cfg["time_horizon"]
    a = cfg["arrival_a"]
    b = cfg["arrival_b"]
    total_expected = float(np.sum(np.exp(a + b * np.arange(1, T + 1) / T)))
    assert total_expected < cfg["num_voters"], (
        f"Total expected arrivals Λ(T) = {total_expected:.2f} must be < N = {cfg['num_voters']}."
    )
    # Same feasibility check for each comparison shape used in plot_bound_over_time.
    for shape in cfg["bound_shapes"]:
        shape_expected = float(np.sum(np.exp(shape["a"] + shape["b"] * np.arange(1, T + 1) / T)))
        assert shape_expected < cfg["num_voters"], (
            f"Shape '{shape['label']}' total expected arrivals Λ(T) = {shape_expected:.2f} "
            f"must be < N = {cfg['num_voters']}."
        )

# Run validation
validate_config(CONFIG)