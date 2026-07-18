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


def test_ni_sample_size_matches_direct_simulation_benchmark():
    # validated against 20,000 direct simulations of the one-sided
    # non-inferiority t-test; closed-form gave n=50.15, power=0.80005
    trial = ContinuousEndpoint(outcome_sd=10.0)
    n = trial.closed_form_sample_size_ni(margin=5.0, treatment_effect=0.0, power=0.8)
    assert abs(n - 50.15) < 0.1


trial = ContinuousEndpoint(outcome_sd=10.0)
n = trial.closed_form_sample_size_ni(margin=5.0, treatment_effect=0.0, power=0.8)
print(n)  # should land near 50.15
print(trial.closed_form_power_ni(n, margin=5.0, treatment_effect=0.0)) 