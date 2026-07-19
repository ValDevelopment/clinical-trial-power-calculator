from power_calc.group_sequential import group_sequential_boundaries, inflation_factor
from power_calc.continuous import ContinuousEndpoint
from power_calc.binary import BinaryEndpoint
from power_calc.survival import SurvivalEndpoint

z_bounds, t, cum_alpha = group_sequential_boundaries(K=5, alpha=0.05)
factor = inflation_factor(z_bounds, t, alpha=0.05, power=0.8)
print(factor)  

trial = ContinuousEndpoint(outcome_sd=10.0)
result = trial.group_sequential_sample_size(treatment_effect=5.0, K=5, power=0.8)
print(result)


trial_b = BinaryEndpoint()
print(trial_b.group_sequential_sample_size(p1=0.30, p2=0.176, K=5, power=0.8))
# n_fixed ~182.3, inflation_factor ~1.04, n_max ~189.6

trial_s = SurvivalEndpoint(accrual_period=6.0, follow_up_period=18.0)
print(trial_s.group_sequential_sample_size(hazard_ratio=0.6, control_median_survival=12.0, K=5, power=0.8))
# n_control ~102.8, events_needed ~125.1, inflation_factor ~1.04