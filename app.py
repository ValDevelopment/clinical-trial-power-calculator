import streamlit as st
import numpy as np
from power_calc.continuous import ContinuousEndpoint
from power_calc.binary import BinaryEndpoint
from power_calc.survival import SurvivalEndpoint
import matplotlib.pyplot as plt

st.title("Clinical Trial Power Calculator")

endpoint_type = st.selectbox(
    "Endpoint type",
    ["Continuous", "Binary", "Survival"],
)

if endpoint_type == "Continuous":
    treatment_effect = st.number_input("Treatment effect (mean difference)", value=5.0)
    outcome_sd = st.number_input("Outcome standard deviation", value=10.0, min_value=0.01)
    baseline_outcome_corr = st.slider("Baseline-outcome correlation", 0.0, 0.95, 0.5)
    power_target = st.slider("Target power", 0.5, 0.99, 0.8)
    alpha = st.number_input("Alpha (significance level)", value=0.05, min_value=0.001, max_value=0.5)

    run_simulation = st.checkbox("Also validate with simulation-based power")
    if run_simulation:
        n_sims = st.number_input("Number of simulations", value=2000, min_value=100, step=100)

    if st.button("Calculate"):
        trial = ContinuousEndpoint(outcome_sd=outcome_sd, baseline_outcome_corr=baseline_outcome_corr, alpha=alpha)
        n_per_arm = trial.closed_form_sample_size(treatment_effect=treatment_effect, power=power_target)
        st.write(f"Required sample size per arm: {n_per_arm:.1f}")
        st.subheader("Power curve")
        n_range = np.linspace(5, n_per_arm * 2, 100)
        power_curve = [trial.closed_form_power(n, treatment_effect) for n in n_range]

        fig, ax = plt.subplots()
        ax.plot(n_range, power_curve)
        ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
        ax.axvline(n_per_arm, color="red", linestyle="--", label=f"required n ({n_per_arm:.1f})")
        ax.set_xlabel("Sample size per arm")
        ax.set_ylabel("Power")
        ax.legend()
        st.pyplot(fig)

        if run_simulation:
            with st.spinner("Running simulation..."):
                sim_power = trial.simulate_power(
                    n_per_arm=round(n_per_arm), treatment_effect=treatment_effect, n_sims=n_sims, seed=1,
                )
            st.write(f"Simulated power at n={round(n_per_arm)}: {sim_power:.3f} (target: {power_target})")
elif endpoint_type == "Binary":
    control_event_rate = st.slider("Control arm event rate", 0.01, 0.99, 0.30)
    odds_ratio = st.number_input("Treatment odds ratio", value=0.5, min_value=0.01)
    power_target = st.slider("Target power", 0.5, 0.99, 0.8)
    alpha = st.number_input("Alpha (significance level)", value=0.05, min_value=0.001, max_value=0.5)

    run_simulation = st.checkbox("Also validate with simulation-based power")
    if run_simulation:
        n_sims = st.number_input("Number of simulations", value=2000, min_value=100, step=100)

    if st.button("Calculate"):
        trial = BinaryEndpoint(alpha=alpha)
        odds_control = control_event_rate / (1 - control_event_rate)
        odds_treatment = odds_control * odds_ratio
        p_treatment = odds_treatment / (1 + odds_treatment)

        n_per_arm = trial.closed_form_sample_size(p1=control_event_rate, p2=p_treatment, power=power_target)
        st.write(f"Treatment arm event rate implied by this odds ratio: {p_treatment:.3f}")
        st.write(f"Required sample size per arm: {n_per_arm:.1f}")
        st.subheader("Power curve")
        n_max = n_per_arm * 2
        n_range = np.linspace(5, n_max, 100)
        power_curve = [trial.closed_form_power(n, control_event_rate, p_treatment) for n in n_range]

        fig, ax = plt.subplots()
        ax.plot(n_range, power_curve)
        ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
        ax.axvline(n_per_arm, color="red", linestyle="--", label=f"required n ({n_per_arm:.1f})")
        ax.set_xlabel("Sample size per arm")
        ax.set_ylabel("Power")
        ax.legend()
        st.pyplot(fig)

        if run_simulation:
            with st.spinner("Running simulation..."):
                sim_power = trial.simulate_power(
                    n_per_arm=round(n_per_arm), control_event_rate=control_event_rate,
                    odds_ratio=odds_ratio, n_sims=n_sims, seed=1,
                )
            st.write(f"Simulated power at n={round(n_per_arm)}: {sim_power:.3f} (target: {power_target})")
elif endpoint_type == "Survival":
    control_median_survival = st.number_input("Control arm median survival", value=12.0, min_value=0.01)
    hazard_ratio = st.number_input("Treatment hazard ratio", value=0.6, min_value=0.01)
    accrual_period = st.number_input("Accrual period", value=6.0, min_value=0.01)
    follow_up_period = st.number_input("Follow-up period (after last enrollment)", value=18.0, min_value=0.01)
    power_target = st.slider("Target power", 0.5, 0.99, 0.8)
    alpha = st.number_input("Alpha (significance level)", value=0.05, min_value=0.001, max_value=0.5)

    run_simulation = st.checkbox("Also validate with simulation-based power")
    if run_simulation:
        n_sims = st.number_input("Number of simulations", value=300, min_value=50, step=50)
        st.caption("Cox model fits are slower than the other endpoints, fewer simulations by default.")

    if st.button("Calculate"):
        trial = SurvivalEndpoint(accrual_period=accrual_period, follow_up_period=follow_up_period, alpha=alpha)
        result = trial.closed_form_sample_size(
            hazard_ratio=hazard_ratio, control_median_survival=control_median_survival, power=power_target,
        )
        st.write(f"Events needed: {result['events_needed']:.1f}")
        st.write(f"Sample size per arm: {result['n_control']:.1f}")
        st.write(f"Total sample size: {result['n_total']:.1f}")
        st.subheader("Power curve")
        n_per_arm_point = result["n_control"]
        n_range = np.linspace(5, n_per_arm_point * 2, 100)
        power_curve = [trial.closed_form_power(n, hazard_ratio, control_median_survival) for n in n_range]

        fig, ax = plt.subplots()
        ax.plot(n_range, power_curve)
        ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
        ax.axvline(n_per_arm_point, color="red", linestyle="--", label=f"required n ({n_per_arm_point:.1f})")
        ax.set_xlabel("Sample size per arm")
        ax.set_ylabel("Power")
        ax.legend()
        st.pyplot(fig)

        if run_simulation:
            with st.spinner("Running simulation (this one's slower)..."):
                sim_power = trial.simulate_power(
                    n_per_arm=round(result["n_control"]), control_median_survival=control_median_survival,
                    hazard_ratio=hazard_ratio, n_sims=n_sims, seed=1,
                )
            st.write(f"Simulated power at n={round(result['n_control'])}: {sim_power:.3f} (target: {power_target})")

