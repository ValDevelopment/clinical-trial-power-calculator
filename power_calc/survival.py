import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from power_calc.group_sequential import group_sequential_boundaries, inflation_factor


class SurvivalEndpoint:
    """
    Time-to-event trial endpoint: data generation (constant-hazard Cox
    model with accrual/censoring), closed-form power/sample size
    (Schoenfeld formula plus event-probability conversion), and
    simulation-based power (Cox PH regression).
    """

    def __init__(self, baseline_mean=50.0, baseline_sd=10.0, baseline_log_hr_coef=0.0,
                 accrual_period=6.0, follow_up_period=18.0, allocation_ratio=1.0, alpha=0.05):
        self.baseline_mean = baseline_mean
        self.baseline_sd = baseline_sd
        self.baseline_log_hr_coef = baseline_log_hr_coef
        self.accrual_period = accrual_period
        self.follow_up_period = follow_up_period
        self.allocation_ratio = allocation_ratio
        self.alpha = alpha

    def generate_trial(self, n_per_arm, control_median_survival, hazard_ratio, dropout_rate=0.0, seed=None):
        rng = np.random.default_rng(seed)
        n_treat = int(round(n_per_arm * self.allocation_ratio))
        n_total = n_per_arm + n_treat
        arm = np.array([0] * n_per_arm + [1] * n_treat)
        baseline = rng.normal(self.baseline_mean, self.baseline_sd, size=n_total)

        h0 = np.log(2) / control_median_survival
        linear_predictor = (np.log(hazard_ratio) * arm
                             + self.baseline_log_hr_coef * (baseline - self.baseline_mean))
        hazard = h0 * np.exp(linear_predictor)
        true_event_time = rng.exponential(1 / hazard)

        enrollment_time = rng.uniform(0, self.accrual_period, size=n_total)
        study_end = self.accrual_period + self.follow_up_period
        admin_censor_time = np.clip(study_end - enrollment_time, 0, None)

        if dropout_rate > 0:
            dropout_hazard = -np.log(1 - dropout_rate) / self.follow_up_period
            dropout_time = rng.exponential(1 / dropout_hazard, size=n_total)
        else:
            dropout_time = np.full(n_total, np.inf)

        observed_time = np.minimum.reduce([true_event_time, admin_censor_time, dropout_time])
        event = (observed_time == true_event_time).astype(int)

        return pd.DataFrame({"subject_id": np.arange(1, n_total + 1), "arm": arm,
                              "baseline": baseline, "time": observed_time, "event": event})

    def closed_form_events_needed(self, hazard_ratio, power=0.8):
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)
        k = self.allocation_ratio
        q = k / (1 + k)
        return (z_alpha + z_beta) ** 2 / (q * (1 - q) * np.log(hazard_ratio) ** 2)

    def _expected_event_probability(self, hazard):
        A, F = self.accrual_period, self.follow_up_period
        T = A + F
        return 1 - (np.exp(-hazard * F) - np.exp(-hazard * T)) / (A * hazard)

    def closed_form_sample_size(self, hazard_ratio, control_median_survival, power=0.8):
        events_needed = self.closed_form_events_needed(hazard_ratio, power)
        h0 = np.log(2) / control_median_survival
        h_treat = h0 * hazard_ratio
        q = self.allocation_ratio / (1 + self.allocation_ratio)

        p_control = self._expected_event_probability(h0)
        p_treat = self._expected_event_probability(h_treat)
        p_overall = q * p_treat + (1 - q) * p_control

        n_total = events_needed / p_overall
        n_control = n_total / (1 + self.allocation_ratio)
        n_treat = n_control * self.allocation_ratio
        return dict(events_needed=events_needed, p_control=p_control, p_treat=p_treat,
                    n_control=n_control, n_treatment=n_treat, n_total=n_control + n_treat)
    
    
    def closed_form_power(self, n_per_arm, hazard_ratio, control_median_survival):
        h0 = np.log(2) / control_median_survival
        h_treat = h0 * hazard_ratio
        q = self.allocation_ratio / (1 + self.allocation_ratio)

        p_control = self._expected_event_probability(h0)
        p_treat = self._expected_event_probability(h_treat)
        p_overall = q * p_treat + (1 - q) * p_control

        n_total = n_per_arm * (1 + self.allocation_ratio)
        events = n_total * p_overall

        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = np.sqrt(events * q * (1 - q)) * abs(np.log(hazard_ratio)) - z_alpha
        return stats.norm.cdf(z_beta)
    
    def closed_form_events_needed_ni(self, margin, true_hazard_ratio=1.0, power=0.8):
        """
        Number of events needed for a one-sided non-inferiority test on
        the hazard-ratio scale. margin is the largest hazard ratio
        (treatment vs. control) still considered non-inferior.
        true_hazard_ratio is the assumed true hazard ratio; 1.0
        (treatment truly equal to control) is the standard conservative
        planning assumption.
        """
        z_alpha = stats.norm.ppf(1 - self.alpha)  # one-sided
        z_beta = stats.norm.ppf(power)
        k = self.allocation_ratio
        q = k / (1 + k)
        log_diff = np.log(margin) - np.log(true_hazard_ratio)
        return (z_alpha + z_beta) ** 2 / (q * (1 - q) * log_diff ** 2)

    def closed_form_sample_size_ni(self, margin, control_median_survival, true_hazard_ratio=1.0, power=0.8):
        events_needed = self.closed_form_events_needed_ni(margin, true_hazard_ratio, power)
        h0 = np.log(2) / control_median_survival
        h_treat = h0 * true_hazard_ratio
        q = self.allocation_ratio / (1 + self.allocation_ratio)

        p_control = self._expected_event_probability(h0)
        p_treat = self._expected_event_probability(h_treat)
        p_overall = q * p_treat + (1 - q) * p_control

        n_total = events_needed / p_overall
        n_control = n_total / (1 + self.allocation_ratio)
        n_treat = n_control * self.allocation_ratio
        return dict(events_needed=events_needed, p_control=p_control, p_treat=p_treat,
                    n_control=n_control, n_treatment=n_treat, n_total=n_control + n_treat)

    def closed_form_power_ni(self, n_per_arm, margin, control_median_survival, true_hazard_ratio=1.0):
        h0 = np.log(2) / control_median_survival
        h_treat = h0 * true_hazard_ratio
        q = self.allocation_ratio / (1 + self.allocation_ratio)

        p_control = self._expected_event_probability(h0)
        p_treat = self._expected_event_probability(h_treat)
        p_overall = q * p_treat + (1 - q) * p_control

        n_total = n_per_arm * (1 + self.allocation_ratio)
        events = n_total * p_overall

        z_alpha = stats.norm.ppf(1 - self.alpha)
        log_diff = np.log(margin) - np.log(true_hazard_ratio)
        z_beta = np.sqrt(events * q * (1 - q)) * abs(log_diff) - z_alpha
        return stats.norm.cdf(z_beta)

    def simulate_power(self, n_per_arm, control_median_survival, hazard_ratio, dropout_rate=0.0,
                        n_sims=500, seed=None):
        rng = np.random.default_rng(seed)
        reject = 0
        for _ in range(n_sims):
            trial_seed = rng.integers(0, 2**31 - 1)
            df = self.generate_trial(n_per_arm, control_median_survival, hazard_ratio, dropout_rate, trial_seed)
            model = PHReg(df["time"], df[["arm", "baseline"]], status=df["event"], ties="efron")
            result = model.fit(disp=0)
            if result.pvalues[0] < self.alpha:
                reject += 1
        return reject / n_sims
    

    def group_sequential_sample_size(self, hazard_ratio, control_median_survival, K, power=0.8):
        fixed = self.closed_form_sample_size(hazard_ratio, control_median_survival, power)
        z_bounds, t, cum_alpha = group_sequential_boundaries(K, alpha=self.alpha)
        factor = inflation_factor(z_bounds, t, alpha=self.alpha, power=power)
        return dict(
            events_needed=fixed["events_needed"] * factor,
            n_control=fixed["n_control"] * factor,
            n_treatment=fixed["n_treatment"] * factor,
            n_total=fixed["n_total"] * factor,
            inflation_factor=factor,
            z_bounds=z_bounds,
            information_fractions=t,
        )
    
