"""
Main entry point for running the optimal stopping model.
"""

import os

from config import CONFIG
from model import VotingModel
from plotting import plot_stopping_boundaries_general, plot_bound_over_time, plot_approx_goodness

def main():
    """
    Run the end-to-end pipeline: build the model, solve it, and render figures.

    Reads parameters from ``CONFIG`` (already validated at import time), solves
    the optimal stopping problem by backward induction, then writes three
    figures: state-space stopping boundaries, the over-time LLR boundary, and
    an exact-versus-approximate threshold comparison.
    """
    cfg = CONFIG
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize the Model
    print("Initializing Voting Model...")
    vm = VotingModel(
        num_voters=cfg["num_voters"],
        time_horizon=cfg["time_horizon"],
        step_cost=cfg["step_cost"],
        reward=cfg["reward"],
        arrival_a=cfg["arrival_a"],
        arrival_b=cfg["arrival_b"]
    )

    # 2. Solve via Backward Induction
    # This computes the Value Function V internally
    vm.solve()

    # 3. Plotting
    asm_track = cfg.get("asm_track", "")
    
    # Plot 1: State Space Boundaries
    plot_stopping_boundaries_general(
        model=vm,
        times=cfg["times_to_plot"],
        mode=cfg["plot_mode"],
        track=asm_track,
        save_path=os.path.join(output_dir, f"stopping_boundaries_{asm_track}_{cfg['plot_mode']}.pdf")
    )
    
    # Plot 2: Time Boundary
    plot_bound_over_time(
        model=vm,
        time_horizon=cfg["time_horizon"],
        bound_shapes=cfg["bound_shapes"],
        save_path=os.path.join(output_dir, "LLR.pdf")
    )

    # Plot 3: Exact versus approximate stopping thresholds
    plot_approx_goodness(
        model=vm,
        times=cfg["times_to_plot"],
        save_path=os.path.join(output_dir, "fig_approx_goodness.pdf")
    )

if __name__ == "__main__":
    main()