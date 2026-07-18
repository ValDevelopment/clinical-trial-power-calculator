import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


class BinaryEndpoint:
    """
    Binary trial endpoint: data generation (logistic model), closed-form
    power/sample size (arcsine / Cohen's h), and simulation-based power
    (logistic regression).
    """

    def __init__(self, baseline_mean=50.0, baseline_sd=10.0, baseline_log_odds_coef=0.0,
                 allocation_ratio=1.0, alpha=0.05):
        self.baseline_mean = baseline_mean
        self.baseline_sd = baseline_sd
        self.baseline_log_odds_coef = baseline_log_odds_coef
        self.allocation_ratio = allocation_ratio
        self.alpha = alpha

    def generate_trial(self, n_per_arm, control_event_rate, odds_ratio, dropout_rate=0.0, seed=None):
        rng = np.random.default_rng(seed)
        n_treat = int(round(n_per_arm * self.allocation_ratio))
        n_total = n_per_arm + n_treat
        arm = np.array([0] * n_per_arm + [1] * n_treat)
        baseline = rng.normal(self.baseline_mean, self.baseline_sd, size=n_total)

        intercept = np.log(control_event_rate / (1 - control_event_rate))
        beta_treatment = np.log(odds_ratio)
        logit_p = (intercept + beta_treatment * arm
                   + self.baseline_log_odds_coef * (baseline - self.baseline_mean))
        p = 1 / (1 + np.exp(-logit_p))
        event = rng.binomial(1, p).astype(float)

        if dropout_rate > 0:
            mask = rng.random(n_total) < dropout_rate
            event = np.where(mask, np.nan, event)

        return pd.DataFrame({"subject_id": np.arange(1, n_total + 1), "arm": arm,
                              "baseline": baseline, "event": event})

    def closed_form_power(self, n_per_arm, p1, p2):
        n1, n2 = n_per_arm, n_per_arm * self.allocation_ratio
        h = cohens_h(p1, p2)
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        ncp = h / np.sqrt(1 / n1 + 1 / n2)
        return stats.norm.sf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)

    def closed_form_sample_size(self, p1, p2, power=0.8):
        def f(n):
            return self.closed_form_power(n, p1, p2) - power
        return brentq(f, 2.0, 1e7)
    
    def closed_form_power_ni(self, n_per_arm, margin, p_control, p_treatment):
        """
        Power for a one-sided non-inferiority test on the risk-difference
        scale. Assumes lower event rates are better (e.g. an adverse
        event or failure endpoint): treatment is declared non-inferior
        if its event rate is not more than `margin` higher than
        control's.

        p_control, p_treatment are the assumed true event rates. Setting
        p_treatment equal to p_control is the standard conservative
        planning assumption (treatment truly equal to control).
        """
        n_c, n_t = n_per_arm, n_per_arm * self.allocation_ratio
        se = np.sqrt(p_treatment * (1 - p_treatment) / n_t + p_control * (1 - p_control) / n_c)
        ncp = (margin + (p_control - p_treatment)) / se
        z_alpha = stats.norm.ppf(1 - self.alpha)
        return stats.norm.cdf(ncp - z_alpha)

    def closed_form_sample_size_ni(self, margin, p_control, p_treatment=None, power=0.8):
        if p_treatment is None:
            p_treatment = p_control  # conservative planning assumption
        def f(n):
            return self.closed_form_power_ni(n, margin, p_control, p_treatment) - power
        return brentq(f, 2.0, 1e7)

    def simulate_power(self, n_per_arm, control_event_rate, odds_ratio, dropout_rate=0.0,
                        n_sims=2000, seed=None):
        rng = np.random.default_rng(seed)
        reject = 0
        for _ in range(n_sims):
            trial_seed = rng.integers(0, 2**31 - 1)
            df = self.generate_trial(n_per_arm, control_event_rate, odds_ratio, dropout_rate, trial_seed)
            df = df.dropna(subset=["event"])
            X = sm.add_constant(df[["arm", "baseline"]])
            model = sm.Logit(df["event"], X).fit(disp=0)
            if model.pvalues["arm"] < self.alpha:
                reject += 1
        return reject / n_sims