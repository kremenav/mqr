# Optimal Stopping Model for Sequential Bayesian Decision-Making

This repository implements the numerical solution of our finite-horizon Bayesian optimal stopping model for dynamic majority decision-making detailed in our paper: "Optimal Dynamic Majority-Quorum Rules: Theory and Evidence" (2026).

The model describes how a collective entity can optimally decide when to stop observing incoming votes and make a final decision (accept or reject) in order to maximize expected utility under uncertainty.

The model formalizes the problem as a dynamic programming process with Poisson-distributed random vote arrivals. At each step, the system decides whether to continue collecting votes — incurring a small observation cost — or to stop and accept/reject based on the posterior belief of the true majority.

The model captures the trade-off between information gain from observing more votes and cost from delaying the decision. In addition to solving for the optimal stopping boundary via backward induction, this implementation provides visualizations of the optimal stopping boundaries, including LLR approximations and optional overlays of the Acceptance–Support Mechanism (ASM) used in our paper.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Model](#mathematical-model)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Visualization](#visualization)
8. [Citation](#citation)

---

## Overview

- The planner observes a population of odd size $N$, where each voter has a fixed binary preference: support (+) or oppose (-) a proposal.
- Votes arrive sequentially over discrete dates $t = 1,2,\dots,T$, in batches of random size drawn from a censored Poisson distribution.
- At each step, the planner chooses to:

    - Stop and Accept
    - Stop and Reject
    - Continue sampling, incurring a small observation cost $\kappa$

- Beliefs are updated via the finite-population hypergeometric posterior.

## Mathematical Model

Let:
- $s_t^+$: cumulative number of “aye” votes observed by time $t$  
- $s_t^-$: cumulative number of “nay” votes observed by time $t$  
- $N$: total number of potential voters  
- $T$: time horizon  
- $\kappa$: step cost per period  
- $R$: reward for a correct decision  

The posterior belief about the number of supporters $M^+$ is 

$$
P(M^+ = m^+ \mid s^+, s^-) = \frac{\binom{m^+}{s^+}\binom{N-m^+}{s^-}}{\binom{N+1}{s+1}}.
$$

Let the value function at time $t$ be $V_t(s^+, s^-)$, representing the maximal expected payoff:

$$
V_t(s^+, s^-) = \max \{ U(s^+, s^-), -\kappa + \mathbb{E}[ V_{t+1}(\tilde{s}^+, \tilde{s}^-) \mid s^+, s^- ] \},
$$

where the stopping payoff is:

$$
U(s^+, s^-) = R \cdot \max \{\mu(s^+, s^-), 1 - \mu(s^+, s^-)\}, \quad
\mu(s^+, s^-) = P(M^+ > N/2 \mid s^+, s^-).
$$

For large $N$ and cumulative sample size $s$, the optimal stopping rule can be approximated via the implemented log-likelihood-ratio (LLR) boundary:

$$
\frac{(s^+ - s^-)^2}{s} \cdot \frac{N}{N-s} \ge 2\left[\log\left(\frac{R}{\kappa}\right) - \log\log\left(\frac{R}{\kappa}\right)\right]\times \frac{N\left[\Lambda(T)-\Lambda(t)\right]}{\left[N-\Lambda(t)\right]\Lambda(T)}.
$$

---

## Repository Structure

```text
mqr/
│
├── asm.py                  # Contains the parameters of ASM tracks
├── config.py               # Configuration of parameters and plotting options
├── model.py                # Bayesian updating and dynamic programming
├── plotting.py             # Visualization of stopping regions and approximations
├── main.py                 # Entry point to run the full model
├── requirements.txt        # Dependencies for Python environment
├── outputs/                # Generated PDF figures
├── .gitignore              # Files excluded from version control
└── README.md               # Project documentation
```

## Installation
Install dependencies:

```
pip install -r requirements.txt
```

## Usage
Run the full model (simulation + visualization):

    python main.py

All generated figures are saved in the dedicated `outputs/` folder.

Typical Workflow:

- Load configuration (`config.py`).
- Initialize `VotingModel'($N$, $T$, $\kappa$, $R$, $\lambda$).
- Compute the value function via vectorized backward induction.
- Plot stopping boundaries using `plot_stopping_boundaries_general()`.
- *(Optional)* Overlay ASM region or LLR approximation.
- Plot over-time LLR boundary using `plot_bound_over_time()`.

## Configuration
Model parameters and plot settings are managed in `config.py`. Validation ensures $N$ is odd, $T \ge 1$, plot times are within the horizon, and the Poisson arrival rate $\lambda$ is feasible.

    CONFIG: Config = {
    "num_voters": 497,
    "time_horizon": 28,
    "step_cost": 0.001,
    "reward": 100.0,
    "arrival_a": 1.7367,
    "arrival_b": 0.0,
    "bound_shapes": [...],
    "plot_mode": "Diff-vs-Sum",
    "asm_track": "",
    "times_to_plot": [2, 14, 26],
}

## Visualization
The system produces three figures:

1. **State-Space Decision Boundaries** (`stopping_boundaries__Diff-vs-Sum.pdf`)

   Displays the adaptive LLR approximation boundary (red curve) and continue region (gray dots) at selected time points ($t = 2, 14, 26$). An optional ASM overlay is shown as a shaded polygon. 
   
   The plotting module supports multiple coordinate transformations:

   | Mode | Description | Axes |
   |------|--------------|------|
   | **Aye-vs-Nay** | Counts of "aye" and "nay" votes.  | $x = s^+$, $y = s^-$ |
   | **Diff-vs-Sum** | Vote margin (difference) against total votes cast. | $x = s^+ + s^-$, $y = s^+ - s^-$ |
   | **Acceptance-vs-Turnout** | Acceptance rate versus overall turnout fraction. | $x = (s^+ + s^-)/N$, $y = s^+/(s^+ + s^-)$ |
   | **Acceptance-vs-Support** | Acceptance rate relative to true underlying support. | $x = s^+/N$, $y = s^+/(s^+ + s^-)$ |

   Select your preferred visualization by setting `plot_mode` in `config.py`.

2. **Posted Boundaries under Front-Loaded Arrivals** (`stopping_boundaries_posted_front_loaded.pdf`)

   Shows the posted LLR approximation under front-loaded arrival processes at selected time points. The gray shaded region represents the "Continue" region between the two boundary branches. The yellow shaded area above the upper curve indicates the ASM region used in the paper.

3. **Adaptive and Posted Boundaries Over Time** (`LLR.pdf`)

   A two-panel comparison figure:
   
   - **(a)** State-space overlay of adaptive (red) and posted (blue) LLR boundaries at early ($t=2$, solid) and late ($t=26$, dashed) times under uniform arrivals, shown in $(s^+ + s^-, s^+ - s^-)$ coordinates.
   
   - **(b)** Over-time visualization of the stopping threshold for the normalized squared vote lead:
     $$\frac{(s^+ - s^-)^2}{s}\,\frac{N}{N-s}$$
     
     Compares three arrival processes: uniform, front-loaded, and back-loaded. The threshold decreases over time, indicating that the planner's stopping criterion becomes more lenient toward the terminal date.

## Citation
If you use this code, methodology, or the resulting plots in your own work, please cite our accompanying paper:

Bhargav Nagaraja Bhatt, Jonas Gehrlein and Kremena Valkanova (2026). *Optimal Dynamic Majority-Quorum Rules: Theory and Evidence*. Working paper.