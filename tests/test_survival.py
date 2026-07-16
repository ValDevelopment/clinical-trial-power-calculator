from power_calc.survival import SurvivalEndpoint

trial = SurvivalEndpoint(accrual_period=6.0, follow_up_period=18.0)
result = trial.closed_form_sample_size(hazard_ratio=0.6, control_median_survival=12.0, power=0.8)
print(result)  

print(trial.simulate_power(n_per_arm=99, control_median_survival=12.0, hazard_ratio=0.6, n_sims=500, seed=1))