from power_calc.binary import BinaryEndpoint

trial = BinaryEndpoint()
n = trial.closed_form_sample_size(p1=0.30, p2=0.176, power=0.8)
print(n)  # ~182.3
print(trial.simulate_power(n_per_arm=182, control_event_rate=0.30, odds_ratio=0.5, n_sims=2000, seed=1))  # expect ~0.78, a bit under 0.80