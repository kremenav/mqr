"""
Plotting utilities for the optimal stopping model.

Provides a coordinate transform between (aye, nay) state space and the various
display modes, a shared publication-quality matplotlib style, and the three main
figures: the per-time stopping boundaries, the over-time LLR boundary, and the
exact-versus-approximate threshold comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import string
from matplotlib.lines import Line2D

# ----------------------------
# Vectorized coordinate transform
# ----------------------------
def transform_coordinates(states, num_voters_total, mode: str = "Diff-vs-Sum"):
    """
    Map (aye, nay) states into the requested display coordinates.

    Parameters
    ----------
    states : sequence of (float, float)
        (aye, nay) points; an empty sequence returns two empty lists.
    num_voters_total : int
        Total number of voters N, used to normalize turnout/support modes.
    mode : str, optional
        One of "Diff-vs-Sum", "Acceptance-vs-Turnout", "Acceptance-vs-Support",
        "Aye-vs-Nay", or "Threshold-vs-Turnout" (default "Diff-vs-Sum").

    Returns
    -------
    tuple
        (xs, ys) arrays in the chosen coordinates.

    Raises
    ------
    ValueError
        If ``mode`` is not recognized.
    """
    if not states:
        return [], []

    states_arr = np.asarray(states)
    aye = states_arr[:, 0]
    nay = states_arr[:, 1]
    total_votes = aye + nay

    if mode == "Diff-vs-Sum":
        return total_votes, aye - nay
    elif mode == "Acceptance-vs-Turnout":
        safe_total = np.where(total_votes > 0, total_votes, 1)
        xs = total_votes / num_voters_total
        ys = np.where(total_votes > 0, aye / safe_total, 0.5)
        return xs, ys
    elif mode == "Acceptance-vs-Support":
        safe_total = np.where(total_votes > 0, total_votes, 1)
        xs = aye / num_voters_total
        ys = np.where(total_votes > 0, aye / safe_total, 0.5)
        return xs, ys
    elif mode == "Aye-vs-Nay":
        return aye, nay
    else:
        raise ValueError(f"Unknown mode {mode}")

# ----------------------------
# Publication-quality matplotlib style
# ----------------------------
def set_econ_style():
    """Apply shared publication-quality matplotlib rcParams (serif, sizes, spines)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 22,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })

# ----------------------------
# Axis formatting helper
# ----------------------------
def format_axes_by_mode(ax, mode, num_voters_total):
    """
    Apply mode-specific reference lines, limits, and axis labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format in place.
    mode : str
        Display mode (see ``transform_coordinates``); unrecognized modes are
        left unformatted.
    num_voters_total : int
        Total number of voters N, used to set limits and reference lines.
    """
    limit = num_voters_total / 2
    ax.yaxis.labelpad = -8

    if mode == "Diff-vs-Sum":
        ax.plot([0, limit], [0, limit], '--', color='#cccccc', linewidth=1, zorder=0)
        ax.plot([0, limit], [0, -limit], '--', color='#cccccc', linewidth=1, zorder=0)
        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
        ax.text(0.95 * limit, 0.5 * limit, "Stop & Accept")
        ax.text(0.95 * limit, -0.5 * limit, "Stop & Reject")
        ax.set(xlim=(0, num_voters_total), ylim=(-limit, limit),
             xlabel=r"$s^+ + s^-$",
             ylabel=r"$s^+ - s^-$")

    elif mode == "Acceptance-vs-Turnout":
        ax.axvline(1.0, color='#cccccc', linestyle='--', linewidth=1, zorder=0)
        ax.text(0.8, 0.9, "Stop & Accept")
        ax.text(0.8, 0.1, "Stop & Reject")
        ax.set(xlim=(0, 1), ylim=(0, 1),
             xlabel=r"$(s^+ + s^-)/N$", ylabel=r"$s^+/(s^+ + s^-)$")

    elif mode == "Acceptance-vs-Support":
        ax.text(0.4, 0.9, "Stop & Accept")
        ax.text(0.4, 0.1, "Stop & Reject")
        ax.set(xlim=(0, 0.5), ylim=(0, 1),
             xlabel=r"$s^+/N$", ylabel=r"$s^+/(s^+ + s^-)$")

    elif mode == "Aye-vs-Nay":
        ax.plot([0, num_voters_total], [num_voters_total, 0], '--', color='#cccccc', linewidth=1, zorder=0)
        ax.text(0.8 * limit, 0.2 * limit, "Stop & Accept")
        ax.text(0.2 * limit, 0.8 * limit, "Stop & Reject")
        ax.set(xlim=(0, limit), ylim=(0, limit),
             xlabel=r"$s^+$", ylabel=r"$s^-$")

    elif mode == "Threshold-vs-Turnout":
        ax.plot([0, limit], [0, limit], '--', color='#cccccc', linewidth=1, zorder=0)
        ax.plot([0, limit], [0, -limit], '--', color='#cccccc', linewidth=1, zorder=0)
        ax.set(xlim=(0, num_voters_total), ylim=(-num_voters_total / 2, num_voters_total / 2),
               xlabel=r"$s^+ + s^- $", ylabel=r"$s^+ - s^-$")

# ----------------------------
# Stopping boundaries plot (per time)
# ----------------------------
def plot_stopping_boundaries_general(model, times, mode="Diff-vs-Sum", track="", save_path=None):
    """
    Plot the stopping boundaries at selected time steps, one subplot per time.

    Each panel shows the continue region (scatter), the closed-form LLR
    approximation, and optionally the shaded ASM region.

    Parameters
    ----------
    model : VotingModel
        A solved model (its ``value_function`` must be populated).
    times : sequence of int
        Time steps to plot; laid out in a single row.
    mode : str, optional
        Display mode passed to ``transform_coordinates`` (default "Diff-vs-Sum").
    track : str, optional
        ASM track to overlay; empty string disables the ASM region (default "").
    save_path : str or None, optional
        If given, save the figure there at 300 dpi; otherwise show it.
    """
    set_econ_style()
    num_voters_total, value_function = model.num_voters_total, model.value_function

    num_plots = len(times)
    cols = num_plots
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, t in enumerate(times):
        ax = axes_flat[idx]

        # 1. Get boundaries
        accept_states, reject_states, continue_states = model.get_boundaries(t)
        xs_cont, ys_cont = transform_coordinates(continue_states, num_voters_total, mode)

        # 2. ASM region (optional)
        if track:
            from asm import compute_ASM_region
            ASM_region = compute_ASM_region(num_voters_total, value_function.shape[0] - 1, t, num_points=200, track=track)
            xs_ASM, ys_ASM = transform_coordinates(ASM_region, num_voters_total, mode)
            if len(xs_ASM):
                ax.fill(xs_ASM, ys_ASM, facecolor='#e0e0e0', edgecolor='#bdbdbd',
                        alpha=0.4, label='Approval-Support Mechanism (ASM)', zorder=1)
                ax.text(xs_ASM[0] + 0.1, ys_ASM[0] + 0.5, "ASM", alpha=0.6)

        # 3. Continue region and adaptive LLR approximation
        ax.scatter(xs_cont, ys_cont, c="#838383", s=15, alpha=0.7, marker='.', label='Continue', zorder=2)
        
        # Adaptive curve (red)
        approx_curve_adaptive = model.simplified(t)
        xs_app_adaptive, ys_app_adaptive = transform_coordinates(approx_curve_adaptive, num_voters_total, mode)
        ax.plot(xs_app_adaptive, ys_app_adaptive, color="#fd0000", linewidth=1.67, alpha=1, label='Adaptive', zorder=3)

        # 4. Format axes
        format_axes_by_mode(ax, mode, num_voters_total)
        ax.set_title(
            rf"$\mathbf{{({string.ascii_lowercase[idx]})}}\quad t={t}$",
            loc='left',
            fontweight='normal',
        )

    # Hide empty axes
    for i in range(num_plots, len(axes_flat)):
        axes_flat[i].set_axis_off()

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_posted_front_loaded_boundaries(
    model, times, arrival_shape, mode="Diff-vs-Sum", track="root", save_path=None
):
    """Plot the posted approximation for a front-loaded arrival profile.

    The ASM overlay uses the same region construction and coordinate transform
    as :func:`plot_stopping_boundaries_general`.
    """
    from model import VotingModel

    set_econ_style()
    front_loaded_model = VotingModel(
        num_voters=model.num_voters_total,
        time_horizon=model.time_horizon,
        step_cost=model.step_cost,
        reward=model.stop_reward,
        arrival_a=arrival_shape["a"],
        arrival_b=arrival_shape["b"],
    )
    num_voters_total = front_loaded_model.num_voters_total
    num_plots = len(times)
    fig, axes = plt.subplots(
        1, num_plots, figsize=(6 * num_plots, 5), constrained_layout=True
    )
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, t in enumerate(times):
        ax = axes_flat[idx]
        posted_curve = front_loaded_model.simplified_posted(t)
        xs_posted, ys_posted = transform_coordinates(
            posted_curve, num_voters_total, mode
        )
        posted_curve_arr = np.asarray(posted_curve)
        midpoint = len(posted_curve_arr) // 2
        posted_x = posted_curve_arr[:midpoint, 0] + posted_curve_arr[:midpoint, 1]
        posted_lead = posted_curve_arr[:midpoint, 0] - posted_curve_arr[:midpoint, 1]
        ax.fill_between(
            posted_x, -posted_lead, posted_lead,
            facecolor="#d9d9d9", alpha=0.45, zorder=1,
        )
        ax.plot(xs_posted, ys_posted, color="#082d47", linewidth=2.0, zorder=2)

        # ASM region: same construction as the stopping-boundaries plot.
        if track:
            from asm import compute_ASM_region
            ASM_region = compute_ASM_region(
                num_voters_total, front_loaded_model.time_horizon, t,
                num_points=200, track=track
            )
            xs_ASM, ys_ASM = transform_coordinates(
                ASM_region, num_voters_total, mode
            )
            if len(xs_ASM):
                ax.fill(
                    xs_ASM, ys_ASM, facecolor="#f2c94c", edgecolor="#c99700",
                    alpha=0.25, zorder=0
                )

        format_axes_by_mode(ax, mode, num_voters_total)
        ax.set_title(
            rf"$\mathbf{{({string.ascii_lowercase[idx]})}}\quad t={t}$",
            loc="left", fontweight="normal",
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

# ----------------------------
# LLR boundary plot (over time)
# ----------------------------
def plot_bound_over_time(model, time_horizon, bound_shapes, save_path=None):
    """
    Plot the over-time stopping boundary for several arrival shapes.

    Draws the right-hand side of Theorem 2 (in the same units as the LHS,
    d²·N/[s(N-s)]) against time for each arrival process in ``bound_shapes``
    (e.g. uniform, front-loaded, back-loaded), each calibrated to the same total
    expected arrivals Λ(T).

    Parameters
    ----------
    model : VotingModel
        Source of N, reward R, and step cost c (its arrival params are not used
        here; the shapes in ``bound_shapes`` drive the curves).
    time_horizon : int
        Time horizon T over which to plot.
    bound_shapes : list of BoundShape
        Arrival shapes to overlay; each a dict with keys ``label``, ``a``,
        ``b``, ``ls``.
    save_path : str or None, optional
        If given, save the figure there at 300 dpi; otherwise show it.
    """
    set_econ_style()
    N = model.num_voters_total
    R = model.stop_reward
    c = model.step_cost
    T = time_horizon

    t_int  = np.arange(T + 1)
    t_grid = np.linspace(0.01, 0.999 * T, 500)

    def boundary(a_val, b_val):
        # Discrete Lambda(t), identical to VotingModel.cumulative_arrivals,
        # so the plot agrees with simplified() and the solver.
        rates = np.zeros(T + 1)
        rates[1:] = np.exp(a_val + b_val * np.arange(1, T + 1) / T)
        Lam = np.cumsum(rates)
        Lam_t = np.interp(t_grid, t_int, Lam)
        Lam_T = Lam[T]
        #ratio = (N - Lam_t) / max(N - Lam_T, 1e-12)
        #gap = np.clip(np.minimum(1.0, np.log(np.maximum(ratio, 1e-12))), 0.0, None)
        ratio = (N * (Lam_T-Lam_t)) / (Lam_T * (N - Lam_t)) 
        return 2.0 * (np.log(R / c) - np.log(np.log(R / c))) * ratio    # squared-LLR units; sqrt before plotting

    fig, (state_ax, ax) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # Panel (a): adaptive and posted boundaries at the early and late dates.
    comparison_times = (2, 26)
    comparison_styles = {2: "-", 26: "--"}
    for t in comparison_times:
        adaptive_curve = model.simplified(t)
        posted_curve = model.simplified_posted(t)
        adaptive_x, adaptive_y = transform_coordinates(adaptive_curve, N, "Diff-vs-Sum")
        posted_x, posted_y = transform_coordinates(posted_curve, N, "Diff-vs-Sum")
        state_ax.plot(
            adaptive_x, adaptive_y, color="#fd0000", linestyle=comparison_styles[t],
            linewidth=1.8,
        )
        state_ax.plot(
            posted_x, posted_y, color="#082d47", linestyle=comparison_styles[t],
            linewidth=1.8,
        )
    format_axes_by_mode(state_ax, "Diff-vs-Sum", N)
    state_ax.set_title(r"$\mathbf{(a)}$", loc="left", fontweight="normal")
    time_handles = [
        Line2D([0], [0], color="black", linestyle=comparison_styles[t],
               linewidth=1.8, label=rf"$t={t}$")
        for t in comparison_times
    ]
    state_ax.legend(
        handles=time_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )

    # Panel (b): the original over-time LLR comparison.
    y_max = 0.0
    for s in bound_shapes:
        y = boundary(s["a"], s["b"])
        ax.plot(t_grid, y, color='k', linestyle=s["ls"], linewidth=1.5,
                label=rf'{s["label"]}')
        y_max = max(y_max, float(np.max(y)))
    ax.set(xlim=(0, T), ylim=(0, y_max * 1.15),
           xlabel="Time ($t$)",
           ylabel=r'$\dfrac{d_t^2}{s_t} \cdot \dfrac{N}{N-s_t}$')

    ax.text(T * 0.5, y_max * 0.95, 'Stop', ha='center', va='center')
    ax.text(T * 0.5, y_max * 0.10, 'Continue', ha='center', va='center')

    ax.legend(loc='upper right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(r"$\mathbf{(b)}$", loc="left", fontweight="normal")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()