from power_calc.binary import BinaryEndpoint

trial = BinaryEndpoint()
n = trial.closed_form_sample_size(p1=0.30, p2=0.176, power=0.8)
print(n)  
print(trial.simulate_power(n_per_arm=182, control_event_rate=0.30, odds_ratio=0.5, n_sims=2000, seed=1)) 

def test_ni_sample_size_matches_direct_simulation_benchmark():
    trial = BinaryEndpoint()
    n = trial.closed_form_sample_size_ni(margin=0.10, p_control=0.20, power=0.8)
    assert abs(n - 197.8) < 0.5