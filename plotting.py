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
        ax.text(0.95 * limit, 0.7 * limit, "Stop & Accept")
        ax.text(0.95 * limit, -0.7 * limit, "Stop & Reject")
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

        # 3. Continue region and LLR approximation
        ax.scatter(xs_cont, ys_cont, c='#4d4d4d', s=15, alpha=0.7, marker='.', label='Continue', zorder=2)
        approx_curve = model.simplified(t)
        xs_app, ys_app = transform_coordinates(approx_curve, num_voters_total, mode)
        ax.plot(xs_app, ys_app, color='#d62728', linewidth=1.6, label='LLR Approx', zorder=3)

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

    fig, ax = plt.subplots(figsize=(8, 5))
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

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

# ----------------------------
# Approximation goodness-of-fit plot
# ----------------------------
def plot_approx_goodness(model, times, save_path=None):
    """Compare exact and simplified thresholds in a two-panel figure."""
    set_econ_style()
    N = model.num_voters_total
    fig, (ax, difference_ax) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = plt.cm.viridis([0.1, 0.40, 0.60])

    for idx, t in enumerate(times):
        _, _, continue_states = model.get_boundaries(t)
        simplified_curve = np.asarray(model.simplified(t))
        simplified_turnout = simplified_curve[:len(simplified_curve) // 2, 0] + simplified_curve[:len(simplified_curve) // 2, 1]
        simplified_lead = simplified_curve[:len(simplified_curve) // 2, 0] - simplified_curve[:len(simplified_curve) // 2, 1]
        continue_by_turnout = {s: set() for s in range(N + 1)}
        for aye, nay in continue_states:
            s = aye + nay
            continue_by_turnout[s].add(abs(aye - nay))

        stop_by_turnout = {s: set() for s in range(N + 1)}
        stop_utilities = model.value_function[model.time_horizon]
        for aye in range(N + 1):
            for nay in range(N - aye + 1):
                if model.value_function[t, aye, nay] <= stop_utilities[aye, nay] + 1e-9:
                    stop_by_turnout[aye + nay].add(abs(aye - nay))

        exact_s, exact_d, approx_s, approx_d = [], [], [], []
        for s in range(N + 1):
            continue_leads = continue_by_turnout[s]
            stop_leads = stop_by_turnout[s]
            if not continue_leads or not stop_leads:
                continue

            all_leads = sorted(continue_leads | stop_leads)
            split_index = len(continue_leads)
            expected_continue = set(all_leads[:split_index])
            if continue_leads != expected_continue or len(all_leads) != len(continue_leads) + len(stop_leads):
                print(f"Warning: interleaved stopping policy at t={t}, s={s}; skipping")
                continue

            d_cont_max = max(continue_leads)
            d_stop_min = min(stop_leads)
            if d_stop_min != d_cont_max + 2:
                raise AssertionError(f"Non-adjacent policy split at t={t}, s={s}")
            exact_s.append(s)
            exact_d.append((d_cont_max + d_stop_min) / 2)
            approx_s.append(s)
            approx_d.append(np.interp(s, simplified_turnout, simplified_lead))

        color = colors[idx % len(colors)]
        ax.plot(exact_s, exact_d, color=color, linestyle='-', linewidth=1.8, label=rf'$t={t}$')
        ax.plot(exact_s, -np.asarray(exact_d), color=color, linestyle='-', linewidth=1.8)
        ax.plot(approx_s, approx_d, color=color, linestyle=':', linewidth=1.8)
        ax.plot(approx_s, -np.asarray(approx_d), color=color, linestyle=':', linewidth=1.8)
        residual = np.asarray(exact_d) - np.asarray(approx_d)
        smoothing_window = 9
        padded_residual = np.pad(
            residual,
            (smoothing_window // 2, smoothing_window // 2),
            mode='edge',
        )
        averaged_residual = np.convolve(
            padded_residual,
            np.ones(smoothing_window) / smoothing_window,
            mode='valid',
        )
        difference_ax.plot(
            exact_s,
            averaged_residual,
            color=color,
            linewidth=1.8,
        )

    format_axes_by_mode(ax, "Threshold-vs-Turnout", N)
    difference_ax.axhline(0, color='#cccccc', linestyle='--', linewidth=1, zorder=0)
    difference_ax.relim()
    difference_ax.autoscale_view()
    residual_limit = max(abs(value) for line in difference_ax.lines for value in line.get_ydata())
    difference_ax.set_ylim(-1.05 * residual_limit, 1.05 * residual_limit)
    difference_ax.set(
        xlim=(0, N),
        xlabel=r"$s^+ + s^-$",
        ylabel=r"$d^* - \tilde{d}$",
    )
    date_handles = [
        Line2D([0], [0], color=colors[idx % len(colors)], linewidth=1.8, label=rf'$t={t}$')
        for idx, t in enumerate(times)
    ]
    fig.legend(
        handles=date_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(date_handles),
        frameon=False,
    )
    ax.set_title(r"$\mathbf{(a)}$", loc='left', fontweight='normal')
    difference_ax.set_title(r"$\mathbf{(b)}$", loc='left', fontweight='normal')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()