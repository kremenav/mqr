import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson
from scipy.special import betaln


class VotingModel:
    """
    Solves the optimal stopping problem for sequential voting.

    Attributes
    ----------
    num_voters_total : int
        Total number of voters.
    time_horizon : int
        Total time horizon for backward induction.
    step_cost : float
        Cost incurred when choosing to continue.
    stop_reward : float
        Reward obtained when stopping.
    arrival_a : float
        Intercept of the log-linear intensity λ(x) = exp(a + b·x), x ∈ [0,1].
    arrival_b : float
        Shape of the log-linear intensity. b<0 = front-loaded, b=0 = uniform,
        b>0 = back-loaded.
    arrival_rates : np.ndarray
        Discrete per-period rates λ_t = exp(a + b·t/T) for t = 1,...,T;
        entry 0 is unused.
    cumulative_arrivals : np.ndarray
        Cumulative expected arrivals Λ(t) = sum_{τ=1}^t λ_τ for t = 0,...,T.
    value_function : np.ndarray
        Value function tensor of shape (time_horizon+1, num_voters_total+1, num_voters_total+1),
        where value_function[t, observed_aye, observed_nay] stores the expected optimal value at time t.
    """

    def __init__(self, num_voters, time_horizon, step_cost, reward, arrival_a, arrival_b):
        # Store parameters
        self.num_voters_total = int(num_voters)
        self.time_horizon = int(time_horizon)
        self.step_cost = float(step_cost)
        self.stop_reward = float(reward)
        self.arrival_a = float(arrival_a)
        self.arrival_b = float(arrival_b)

        # Per-period rates from the log-linear intensity λ(x) = exp(a + b·x),
        # discretized at x = t/T for t = 1,...,T. Index 0 is unused so that
        # arrival_rates[t] corresponds to λ_t in the paper's notation.
        T = self.time_horizon
        self.arrival_rates = np.zeros(T + 1, dtype=np.float64)
        self.arrival_rates[1:] = np.exp(
            self.arrival_a + self.arrival_b * np.arange(1, T + 1) / T
        )

        # Cumulative expected arrivals Λ(t) = sum_{τ=1}^t λ_τ; Λ(0) = 0.
        self.cumulative_arrivals = np.zeros(T + 1, dtype=np.float64)
        self.cumulative_arrivals[1:] = np.cumsum(self.arrival_rates[1:])

        # Initialize value function tensor: invalid states = -inf
        self.value_function = np.full(
            (self.time_horizon + 1, self.num_voters_total + 1, self.num_voters_total + 1),
            -np.inf,
            dtype=np.float64
        )

        # Precompute log-factorials for combinatorial calculations
        self._log_factorials = gammaln(np.arange(self.num_voters_total + 3))

        # Caches for speed
        self._posterior_cache = {}        # Stores posterior probabilities per total observed votes
        self._predictive_cache = {}       # Stores predictive probabilities per (total_votes_observed, batch_size)
        self._terminal_utilities = None   # Terminal stop utility matrix

    # ----------------------------
    # Combinatorial helper
    # ----------------------------
    def _log_combination(self, n, k):
        """
        Vectorized log of the binomial coefficient, log(n choose k).

        Broadcasts ``n`` and ``k`` together and returns -inf wherever k is
        out of range (k < 0 or k > n), so invalid states drop out of any
        downstream log-sum-exp.

        Parameters
        ----------
        n, k : int or array_like
            Operands of (n choose k); broadcast against each other.

        Returns
        -------
        np.ndarray
            log(n choose k), with -inf at infeasible (n, k).
        """
        n_broadcast, k_broadcast = np.broadcast_arrays(n, k)
        valid_mask = (k_broadcast >= 0) & (k_broadcast <= n_broadcast)
        n_safe = np.where(valid_mask, n_broadcast, 0)
        k_safe = np.where(valid_mask, k_broadcast, 0)
        log_comb_values = (
            self._log_factorials[n_safe + 1]
            - self._log_factorials[k_safe + 1]
            - self._log_factorials[n_safe - k_safe + 1]
        )
        return np.where(valid_mask, log_comb_values, -np.inf)

    # ----------------------------
    # Posterior probability layer
    # ----------------------------
    def _posterior_layer(self, total_votes_observed):
        """
        Posterior over the true aye count for every state at a given turnout.

        For all states sharing ``total_votes_observed`` total votes, computes
        P(true_aye_count | observed_aye, observed_nay) under a uniform prior on
        the true aye count, via a log-sum-exp normalization. Results are cached
        per ``total_votes_observed``.

        Parameters
        ----------
        total_votes_observed : int
            Total observed votes s+ + s- shared by the states in this layer.

        Returns
        -------
        np.ndarray
            Array of shape (num_voters_total+1, total_votes_observed+1). Row m is
            the true aye count; column j indexes the state
            (observed_aye=j, observed_nay=total_votes_observed-j). Columns are
            normalized to sum to 1.
        """
        if total_votes_observed in self._posterior_cache:
            return self._posterior_cache[total_votes_observed]

        N = self.num_voters_total
        true_aye_counts = np.arange(N + 1)                     # Possible true aye votes
        observed_aye_counts = np.arange(total_votes_observed + 1)
        observed_nay_counts = total_votes_observed - observed_aye_counts

        # Log-posterior up to proportionality
        log_posterior = (
            self._log_combination(true_aye_counts[:, None], observed_aye_counts[None, :])
            + self._log_combination((N - true_aye_counts)[:, None], observed_nay_counts[None, :])
        )

        # Mask infeasible true counts
        feasible_mask = (true_aye_counts[:, None] >= observed_aye_counts[None, :]) & \
                        (true_aye_counts[:, None] <= (N - observed_nay_counts)[None, :])
        log_posterior[~feasible_mask] = -np.inf

        # Normalize using log-sum-exp trick
        max_per_column = np.max(log_posterior, axis=0, keepdims=True)
        posterior_probs = np.exp(log_posterior - max_per_column)
        posterior_probs /= np.sum(posterior_probs, axis=0, keepdims=True)

        self._posterior_cache[total_votes_observed] = posterior_probs
        return posterior_probs

    # ----------------------------
    # Terminal utilities
    # ----------------------------
    def compute_terminal_utilities(self):
        """
        Stop utility for every state (the terminal-date payoff).

        For each state the decision maker picks the more likely outcome under
        the posterior, earning ``stop_reward`` times the probability of being
        correct:
        U = stop_reward · max(P(aye-majority | state), 1 - P(aye-majority | state)).
        The result is cached on first call.

        Returns
        -------
        np.ndarray
            Array of shape (N+1, N+1) indexed by (observed_aye, observed_nay);
            infeasible states (s+ + s- > N) remain -inf.
        """
        if self._terminal_utilities is not None:
            return self._terminal_utilities

        N = self.num_voters_total
        stop_utilities = np.full((N + 1, N + 1), -np.inf, dtype=np.float64)
        aye_majority_threshold = (N // 2) + 1

        for total_votes_observed in range(N + 1):
            observed_aye_counts = np.arange(total_votes_observed + 1)
            posterior_probs = self._posterior_layer(total_votes_observed)
            p_aye_majority = np.sum(posterior_probs[aye_majority_threshold:, :], axis=0)
            stop_utilities[observed_aye_counts, total_votes_observed - observed_aye_counts] = \
                self.stop_reward * np.maximum(p_aye_majority, 1.0 - p_aye_majority)

        self._terminal_utilities = stop_utilities
        return stop_utilities

    # ----------------------------
    # Predictive probabilities
    # ----------------------------
    def _predictive_layer(self, total_votes_observed, batch_size):
        """
        Predictive distribution of ayes in the next batch, per state.

        Given a state at turnout ``total_votes_observed`` and a batch of
        ``batch_size`` new votes, returns P(batch_aye = k | state) for
        k = 0,...,batch_size. Uses the Beta-Binomial closed form implied by the
        uniform prior (equivalently, marginalizing the posterior of
        ``_posterior_layer``). Results are cached per (turnout, batch_size).

        Parameters
        ----------
        total_votes_observed : int
            Total observed votes s+ + s- before the batch.
        batch_size : int
            Number of new votes in the batch.

        Returns
        -------
        np.ndarray
            Array of shape (batch_size+1, total_votes_observed+1). Row k is the
            number of ayes in the batch; column j indexes the prior state
            (observed_aye=j, observed_nay=total_votes_observed-j).
        """
        key = (total_votes_observed, batch_size)
        if key in self._predictive_cache:
            return self._predictive_cache[key]
        tvo, bs = total_votes_observed, batch_size
        oac = np.arange(tvo + 1)          # s+ for each state
        onc = tvo - oac                   # s-
        k = np.arange(bs + 1)[:, None]    # batch ayes a+
        # BetaBinomial(bs, s+ +1, s- +1):  C(bs,k) * B(k+s+ +1, bs-k+s- +1) / B(s+ +1, s- +1)
        log_choose = self._log_factorials[bs + 1] - self._log_factorials[k + 1] - self._log_factorials[bs - k + 1]
        pr = np.exp(log_choose
                + betaln(k + oac[None, :] + 1, (bs - k) + onc[None, :] + 1)
                - betaln(oac + 1, onc + 1)[None, :])
        self._predictive_cache[key] = pr
        return pr

    # ----------------------------
    # Backward induction solver
    # ----------------------------
    def solve(self):
        """
        Fill the value function by backward induction over the horizon.

        Initializes the terminal layer with the stop utilities, then sweeps
        t = T-1,...,0. At each state the Bellman update compares stopping now
        against continuing: continuing pays -step_cost plus the expected future
        value, where the next batch size is Poisson(λ_{t+1}) (truncated to the
        remaining votes) and the batch composition follows ``_predictive_layer``.

        Returns
        -------
        np.ndarray
            The filled ``value_function`` tensor (also stored on the instance).
        """
        N, T = self.num_voters_total, self.time_horizon
        stop_utilities = self.compute_terminal_utilities()
        self.value_function[T] = stop_utilities.copy()

        for t in range(T - 1, -1, -1):
            self.value_function[t] = stop_utilities.copy()  # Default: stop

            # Rate governing arrivals between value-function levels t and t+1.
            lambda_next = self.arrival_rates[t + 1]

            for total_votes_observed in range(N):
                remaining_votes = N - total_votes_observed
                batch_sizes = np.arange(remaining_votes)
                poisson_probs = poisson.pmf(batch_sizes, lambda_next)
                poisson_censored_prob = 1.0 - np.sum(poisson_probs)
                batch_probabilities = np.append(poisson_probs, poisson_censored_prob)
                valid_batch_indices = np.where(batch_probabilities > 1e-9)[0]
                if len(valid_batch_indices) == 0:
                    continue

                observed_aye_counts = np.arange(total_votes_observed + 1)
                observed_nay_counts = total_votes_observed - observed_aye_counts
                expected_future_value = np.zeros_like(observed_aye_counts, dtype=np.float64)

                for batch_size in valid_batch_indices:
                    prob_batch = batch_probabilities[batch_size]
                    if batch_size == 0:
                        expected_future_value += prob_batch * self.value_function[t + 1, observed_aye_counts, observed_nay_counts]
                        continue

                    predictive_probs = self._predictive_layer(total_votes_observed, batch_size)
                    next_batch_aye_counts = np.arange(batch_size + 1)
                    next_state_aye_indices = observed_aye_counts[None, :] + next_batch_aye_counts[:, None]
                    next_state_nay_indices = observed_nay_counts[None, :] + (batch_size - next_batch_aye_counts[:, None])
                    future_values_next_state = self.value_function[t + 1, next_state_aye_indices, next_state_nay_indices]
                    expected_future_value += prob_batch * np.sum(predictive_probs * future_values_next_state, axis=0)

                # Bellman update: stop vs continue
                self.value_function[t, observed_aye_counts, observed_nay_counts] = \
                    np.maximum(stop_utilities[observed_aye_counts, observed_nay_counts],
                               -self.step_cost + expected_future_value)

        return self.value_function

    # ----------------------------
    # Extract stopping boundaries
    # ----------------------------
    def get_boundaries(self, t):
        """
        Classify every feasible state at time t as accept, reject, or continue.

        A state is a stop state when its value equals the stop utility (within a
        tolerance); stop states are split into accept (aye > nay) and reject
        (nay > aye), with ties treated as wait. All other states continue.

        Parameters
        ----------
        t : int
            Time step at which to read the (already solved) value function.

        Returns
        -------
        tuple of list
            (accept_states, reject_states, wait_states), each a list of
            (observed_aye, observed_nay) tuples.
        """
        accept_states, reject_states, wait_states = [], [], []
        stop_utilities = self.value_function[self.time_horizon]

        for observed_aye in range(self.num_voters_total + 1):
            for observed_nay in range(self.num_voters_total - observed_aye + 1):
                current_value = self.value_function[t, observed_aye, observed_nay]
                stop_utility_value = stop_utilities[observed_aye, observed_nay]

                if current_value <= stop_utility_value + 1e-9:
                    if observed_aye > observed_nay:
                        accept_states.append((observed_aye, observed_nay))
                    elif observed_nay > observed_aye:
                        reject_states.append((observed_aye, observed_nay))
                    else:
                        wait_states.append((observed_aye, observed_nay))
                else:
                    wait_states.append((observed_aye, observed_nay))

        return accept_states, reject_states, wait_states

    # ----------------------------
    # Simplified approximate LLR boundary - adaptive
    # ----------------------------
    def simplified(self, t: int):
        """
        Approximate (closed-form) stopping boundary at time t.

        Evaluates the log-likelihood-ratio approximation of Theorem 2 across a
        range of total votes s, giving the boundary half-width
        d = sqrt(2·[log(R/c) - log(log(R/c))] ·
        N(Λ(T)-Λ(t))/[(N-Λ(t))Λ(T)] · s(N-s)/N),
        and returns both branches as (aye, nay) = ((s±d)/2, (s∓d)/2).

        Parameters
        ----------
        t : int
            Time step at which to evaluate the boundary.

        Returns
        -------
        list of (float, float)
            Boundary points: the upper branch followed by the lower branch in
            reverse, so the sequence traces a single continuous curve.
        """
        N, R, c, T = self.num_voters_total, self.stop_reward, self.step_cost, self.time_horizon
        # Cumulative expected arrivals at the current date and at the deadline.
        Lambda_t = self.cumulative_arrivals[t]
        Lambda_T = self.cumulative_arrivals[T]

        upper_branch_states, lower_branch_states = [], []

        total_votes_range = np.linspace(0, N, 300)

        reward_cost_log_ratio = (np.log(R / c) -np.log(np.log(R / c)))
        # Information-time gap:
        #info_time_gap = (N* (Lambda_T-Lambda_t)) / ((Lambda_T)*(N-Lambda_t)) 
        

        for total_votes_observed in total_votes_range:
            # d² = [2log(R/c) − 2loglog(R/c)] · N(Λ_T−Λ_t)/((N−Λ_t)Λ_T) · s(N−s)/N
            s_hat_T = min(total_votes_observed + Lambda_T - Lambda_t, N)
            llr_term = reward_cost_log_ratio * 2 * total_votes_observed * (s_hat_T - total_votes_observed) / (s_hat_T)
            # llr_term = reward_cost_log_ratio * info_time_gap * (2 * total_votes_observed * (N - total_votes_observed)) / N
            llr_sqrt_term = np.sqrt(max(0, llr_term))
            # EDIT 4: skip infeasible states (boundary outside the simplex)
            #if llr_sqrt_term > total_votes_observed:
            #   continue

            upper_aye = (total_votes_observed + llr_sqrt_term) / 2
            upper_nay = (total_votes_observed - llr_sqrt_term) / 2
            upper_branch_states.append((upper_aye, upper_nay))

            lower_aye = (total_votes_observed - llr_sqrt_term) / 2
            lower_nay = (total_votes_observed + llr_sqrt_term) / 2
            lower_branch_states.append((lower_aye, lower_nay))

        return upper_branch_states + lower_branch_states[::-1]
    
    # ----------------------------
    # Simplified approximate LLR boundary - posted
    # ----------------------------
    
    def simplified_posted(self, t: int):
            """
            Approximate (closed-form) stopping boundary at time t.
    
            Evaluates the log-likelihood-ratio approximation of Theorem 2 across a
            range of total votes s, giving the boundary half-width
            d = sqrt(2·[log(R/c) - log(log(R/c))] ·
            N(Λ(T)-Λ(t))/[(N-Λ(t))Λ(T)] · s(N-s)/N),
            and returns both branches as (aye, nay) = ((s±d)/2, (s∓d)/2).
    
            Parameters
            ----------
            t : int
                Time step at which to evaluate the boundary.
    
            Returns
            -------
            list of (float, float)
                Boundary points: the upper branch followed by the lower branch in
                reverse, so the sequence traces a single continuous curve.
            """
            N, R, c, T = self.num_voters_total, self.stop_reward, self.step_cost, self.time_horizon
            # Cumulative expected arrivals at the current date and at the deadline.
            Lambda_t = self.cumulative_arrivals[t]
            Lambda_T = self.cumulative_arrivals[T]
    
            upper_branch_states, lower_branch_states = [], []
    
            total_votes_range = np.linspace(0, N, 300)
    
            reward_cost_log_ratio = (np.log(R / c) -np.log(np.log(R / c)))
            # Information-time gap:
            info_time_gap = (N* (Lambda_T-Lambda_t)) / ((Lambda_T)*(N-Lambda_t)) 
            
    
            for total_votes_observed in total_votes_range:
                # d² = [2log(R/c) − 2loglog(R/c)] · N(Λ_T−Λ_t)/((N−Λ_t)Λ_T) · s(N−s)/N
                llr_term = reward_cost_log_ratio * info_time_gap * (2 * total_votes_observed * (N - total_votes_observed)) / N
                llr_sqrt_term = np.sqrt(max(0, llr_term))

    
                upper_aye = (total_votes_observed + llr_sqrt_term) / 2
                upper_nay = (total_votes_observed - llr_sqrt_term) / 2
                upper_branch_states.append((upper_aye, upper_nay))
    
                lower_aye = (total_votes_observed - llr_sqrt_term) / 2
                lower_nay = (total_votes_observed + llr_sqrt_term) / 2
                lower_branch_states.append((lower_aye, lower_nay))
    
            return upper_branch_states + lower_branch_states[::-1]