from power_calc.survival import SurvivalEndpoint

trial = SurvivalEndpoint(accrual_period=6.0, follow_up_period=18.0)
result = trial.closed_form_sample_size(hazard_ratio=0.6, control_median_survival=12.0, power=0.8)
print(result)  

print(trial.simulate_power(n_per_arm=99, control_median_survival=12.0, hazard_ratio=0.6, n_sims=500, seed=1))

def test_closed_form_power_roundtrips_sample_size():
    trial = SurvivalEndpoint(accrual_period=6.0, follow_up_period=18.0)
    result = trial.closed_form_sample_size(hazard_ratio=0.6, control_median_survival=12.0, power=0.8)
    power = trial.closed_form_power(result["n_control"], hazard_ratio=0.6, control_median_survival=12.0)
    assert abs(power - 0.8) < 1e-6

def test_ni_sample_size_roundtrips_to_target_power():
    trial = SurvivalEndpoint(accrual_period=6.0, follow_up_period=18.0)
    result = trial.closed_form_sample_size_ni(margin=1.3, control_median_survival=12.0, power=0.8)
    power = trial.closed_form_power_ni(result["n_control"], margin=1.3, control_median_survival=12.0)
    assert abs(power - 0.8) < 1e-6