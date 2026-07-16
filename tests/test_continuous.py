from power_calc.continuous import ContinuousEndpoint


def test_closed_form_sample_size_matches_known_benchmark():
    # Cohen's d = 0.5, alpha 0.05, power 0.8 
    trial = ContinuousEndpoint()
    n = trial.closed_form_sample_size(treatment_effect=5.0, power=0.8)
    assert abs(n - 63.77) < 0.1


def test_simulated_power_matches_closed_form_with_no_baseline_effect():
    trial = ContinuousEndpoint(baseline_outcome_corr=0.0)
    simulated = trial.simulate_power(n_per_arm=64, treatment_effect=5.0, n_sims=2000, seed=1)
    closed_form = trial.closed_form_power(n_per_arm=64, treatment_effect=5.0)
    assert abs(simulated - closed_form) < 0.03  # simulation noise tolerance



trial = ContinuousEndpoint()
print(trial.closed_form_sample_size(treatment_effect=5.0, power=0.8))  
print(trial.simulate_power(n_per_arm=64, treatment_effect=5.0, n_sims=2000, seed=1))  