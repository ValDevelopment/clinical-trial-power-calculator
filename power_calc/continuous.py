import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import statsmodels.api as sm
from power_calc.group_sequential import group_sequential_boundaries, inflation_factor


class ContinuousEndpoint:
    """
    Continuous trial endpoint: data generation, closed-form power/sample
    size (noncentral t), and simulation-based power (ANCOVA).
    """

    def __init__(self, baseline_mean=50.0, baseline_sd=10.0,
                 baseline_outcome_corr=0.5, outcome_sd=10.0,
                 allocation_ratio=1.0, alpha=0.05):
        self.baseline_mean = baseline_mean
        self.baseline_sd = baseline_sd
        self.baseline_outcome_corr = baseline_outcome_corr
        self.outcome_sd = outcome_sd
        self.allocation_ratio = allocation_ratio
        self.alpha = alpha

    def generate_trial(self, n_per_arm, treatment_effect, dropout_rate=0.0, seed=None):
        rng = np.random.default_rng(seed)
        n_treat = int(round(n_per_arm * self.allocation_ratio))
        n_total = n_per_arm + n_treat
        arm = np.array([0] * n_per_arm + [1] * n_treat)
        baseline = rng.normal(self.baseline_mean, self.baseline_sd, size=n_total)

        beta_baseline = self.baseline_outcome_corr * (self.outcome_sd / self.baseline_sd)
        residual_sd = self.outcome_sd * np.sqrt(1 - self.baseline_outcome_corr**2)
        outcome_mean = (self.baseline_mean
                        + beta_baseline * (baseline - self.baseline_mean)
                        + treatment_effect * arm)
        outcome = outcome_mean + rng.normal(0, residual_sd, size=n_total)

        if dropout_rate > 0:
            mask = rng.random(n_total) < dropout_rate
            outcome = np.where(mask, np.nan, outcome)

        return pd.DataFrame({"subject_id": np.arange(1, n_total + 1), "arm": arm,
                              "baseline": baseline, "outcome": outcome})

    def closed_form_power(self, n_per_arm, treatment_effect):
        n1, n2 = n_per_arm, n_per_arm * self.allocation_ratio
        df = n1 + n2 - 2
        ncp = treatment_effect / (self.outcome_sd * np.sqrt(1 / n1 + 1 / n2))
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        return 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

    def closed_form_sample_size(self, treatment_effect, power=0.8):
        def f(n):
            return self.closed_form_power(n, treatment_effect) - power
        lo = 2.0
        if f(lo) >= 0:
            return lo
        hi = 4.0
        f_hi = f(hi)
        while not (np.isfinite(f_hi) and f_hi > 0):
            hi *= 1.5 if np.isfinite(f_hi) else 1.05
            f_hi = f(hi)
            if hi > 1e7:
                raise RuntimeError("could not bracket a solution, check inputs")
        return brentq(f, lo, hi)
    
    def closed_form_power_ni(self, n_per_arm, margin, treatment_effect=0.0):
        """
        Power for a one-sided non-inferiority test at a fixed per-arm
        sample size. treatment_effect is the assumed true mean
        difference (treatment minus control), 0 being the standard
        conservative planning assumption. margin is the largest
        acceptable amount treatment could be worse than control and
        still be declared non-inferior.
        """
        n1, n2 = n_per_arm, n_per_arm * self.allocation_ratio
        df = n1 + n2 - 2
        ncp = (treatment_effect + margin) / (self.outcome_sd * np.sqrt(1 / n1 + 1 / n2))
        t_crit = stats.t.ppf(1 - self.alpha, df)  # one-sided
        return 1 - stats.nct.cdf(t_crit, df, ncp)

    def closed_form_sample_size_ni(self, margin, treatment_effect=0.0, power=0.8):
        def f(n):
            return self.closed_form_power_ni(n, margin, treatment_effect) - power
        lo = 2.0
        if f(lo) >= 0:
            return lo
        hi = 4.0
        f_hi = f(hi)
        while not (np.isfinite(f_hi) and f_hi > 0):
            hi *= 1.5 if np.isfinite(f_hi) else 1.05
            f_hi = f(hi)
            if hi > 1e7:
                raise RuntimeError("could not bracket a solution, check inputs")
        return brentq(f, lo, hi)

    def simulate_power(self, n_per_arm, treatment_effect, dropout_rate=0.0, n_sims=2000, seed=None):
        rng = np.random.default_rng(seed)
        reject = 0
        for _ in range(n_sims):
            trial_seed = rng.integers(0, 2**31 - 1)
            df = self.generate_trial(n_per_arm, treatment_effect, dropout_rate, trial_seed)
            df = df.dropna(subset=["outcome"])
            X = sm.add_constant(df[["arm", "baseline"]])
            model = sm.OLS(df["outcome"], X).fit()
            if model.pvalues["arm"] < self.alpha:
                reject += 1
        return reject / n_sims
    

    def group_sequential_sample_size(self, treatment_effect, K, power=0.8):
        """
        Maximum per-arm sample size for a group sequential design with K
        equally spaced O'Brien-Fleming interim looks.
        """
        n_fixed = self.closed_form_sample_size(treatment_effect, power)
        z_bounds, t, cum_alpha = group_sequential_boundaries(K, alpha=self.alpha)
        factor = inflation_factor(z_bounds, t, alpha=self.alpha, power=power)
        return dict(
            n_fixed=n_fixed,
            inflation_factor=factor,
            n_max=n_fixed * factor,
            z_bounds=z_bounds,
            information_fractions=t,
        )