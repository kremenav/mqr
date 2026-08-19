"""
Approval-Support Mechanism (ASM) region.

Constructs the (aye, nay) region carved out by an ASM rule, parameterized by a
time-varying approval threshold and support (turnout) threshold. The chosen
``track`` selects which pair of threshold curves to use.
"""

import numpy as np


# ----------------------------
# ASM region construction
# ----------------------------
def compute_ASM_region(num_voters, time_horizon, t, num_points=200, track="root"):
    """
    Build the (aye, nay) polygon of the ASM acceptance region at time t.

    The ASM rule is defined by two thresholds that vary with t/time_horizon:
    an ``approval`` threshold on the aye-share s+/(s+ + s-) and a ``support``
    threshold on turnout s+/N. The selected ``track`` picks the threshold
    curves. The returned polygon spans aye in [support·N, N], with its upper
    edge tracing the approval threshold and its lower edge running along nay = 0.

    Parameters
    ----------
    num_voters : int
        Total number of voters N.
    time_horizon : int
        Time horizon T used to normalize t into [0, 1].
    t : int
        Current time step; only the ratio t / time_horizon is used.
    num_points : int, optional
        Number of samples along the aye axis (default 200).
    track : str, optional
        Threshold-curve family: "default", "root", "big", "medium", or "small"
        (default "root").

    Returns
    -------
    list of (float, float)
        (aye, nay) vertices of the region polygon: the upper (approval-threshold)
        edge followed by the lower (nay = 0) edge in reverse, forming a closed loop.

    Raises
    ------
    ValueError
        If ``track`` is not one of the recognized families.
    """
    # Select the approval/support threshold curves for the requested track.
    if track == "default":
        support = 0.5 - t / (2 * time_horizon)
        approval = 1 / (t / time_horizon + 1)

    elif track == "root":
        approval = 0.10714275 / (t / time_horizon + 0.2142855) + 0.5
        support = 0.5 - 0.5 * t / time_horizon

    elif track == "big":
        approval = 1.0 - 0.5 * min(t / time_horizon / (23/28), 1.0)
        support = 0.00583 / (t / time_horizon + 0.01166)

    elif track == "medium":
        approval = 1.0 - 0.5 * min(t / time_horizon / (17/28), 1.0)
        support = 0.004375 / (t / time_horizon + 0.00875)

    elif track == "small":
        approval = 1.0 - 0.5 * min(t / time_horizon / (10/28), 1.0)
        support = 0.002915 / (t / time_horizon + 0.00583)

    else:
        raise ValueError(f"Unknown track '{track}'")

    # Map thresholds into (aye, nay) space.
    # Upper edge: at each aye level, nay sits on the approval threshold, i.e.
    # approval = aye / (aye + nay)  =>  nay = (1 - approval)·aye / approval.
    aye_min = support * num_voters
    aye_vals = np.linspace(aye_min, num_voters, num_points)
    nay_max = (1 - approval) * aye_vals / approval

    # Close the polygon: upper edge, then back along the nay = 0 axis.
    points = list(zip(aye_vals, nay_max))
    points += list(zip(aye_vals[::-1], np.zeros_like(aye_vals)))
    return points
