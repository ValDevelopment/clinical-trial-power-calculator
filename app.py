import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from power_calc.continuous import ContinuousEndpoint
from power_calc.binary import BinaryEndpoint
from power_calc.survival import SurvivalEndpoint

st.title("Clinical Trial Power Calculator")

st.markdown(
    """
    Estimate the sample size needed for a two-arm clinical trial, or the
    statistical power a given sample size provides. Choose an endpoint
    type below, fill in your study assumptions, and click **Calculate**.
    """
)

endpoint_type = st.selectbox(
    "Endpoint type",
    ["Continuous", "Binary", "Survival"],
)

if endpoint_type == "Continuous":
    test_type = st.radio(
        "Test type",
        ["Superiority", "Non-inferiority"],
        help="Superiority tests whether the treatment differs from control. Non-inferiority tests whether the treatment is not worse than control by more than a specified margin.",
    )

    if test_type == "Superiority":
        treatment_effect = st.number_input(
            "Treatment effect (mean difference)", value=5.0,
            help="True mean difference in outcome between the treatment and control arms.",
        )
        outcome_sd = st.number_input(
            "Outcome standard deviation", value=10.0, min_value=0.01,
            help="Natural variability in outcome between subjects, independent of treatment.",
        )
        baseline_outcome_corr = st.slider(
            "Baseline-outcome correlation", 0.0, 0.95, 0.5,
            help="Correlation between baseline value and final outcome. Higher values increase the power gained from adjusting for baseline in the analysis.",
        )
        power_target = st.slider(
            "Target power", 0.5, 0.99, 0.8,
            help="Probability of detecting the treatment effect at the calculated sample size, assuming the effect is real.",
        )
        alpha = st.number_input(
            "Alpha (significance level)", value=0.05, min_value=0.001, max_value=0.5,
            help="Acceptable false-positive rate. Conventional default is 0.05.",
        )
        allocation_ratio = st.number_input(
            "Allocation ratio (treatment:control)", value=1.0, min_value=0.1,
            help="Ratio of treatment arm size to control arm size. A value of 1.0 indicates equal allocation. A value of 2.0 indicates twice as many subjects in the treatment arm as the control arm.",
        )

        run_simulation = st.checkbox(
            "Also validate with simulation-based power",
            help="Generates synthetic trial data and fits the ANCOVA model directly to estimate power. Slower than the closed-form formula. Reflects the result of an actual regression-based analysis.",
        )
        if run_simulation:
            n_sims = st.number_input("Number of simulations", value=2000, min_value=100, step=100)

        if st.button("Calculate"):
            if treatment_effect == 0:
                st.error("Treatment effect is 0. No finite sample size exists when there is no difference between arms. Enter a non-zero treatment effect.")
            else:
                trial = ContinuousEndpoint(
                    outcome_sd=outcome_sd, baseline_outcome_corr=baseline_outcome_corr,
                    alpha=alpha, allocation_ratio=allocation_ratio,
                )
                try:
                    n_control = trial.closed_form_sample_size(treatment_effect=treatment_effect, power=power_target)
                except RuntimeError:
                    st.error(
                        "The required sample size exceeds 10 million subjects per arm, far beyond any "
                        "realistic trial. This treatment effect is too small relative to the outcome "
                        "standard deviation to be practically detectable. Consider a larger effect or a "
                        "smaller standard deviation."
                    )
                else:
                    n_treatment = n_control * allocation_ratio
                    st.write(f"Control arm sample size: {n_control:.1f}")
                    st.write(f"Treatment arm sample size: {n_treatment:.1f}")
                    st.write(f"Total sample size: {n_control + n_treatment:.1f}")

                    st.subheader("Power curve")
                    n_range = np.linspace(5, n_control * 2, 100)
                    power_curve = [trial.closed_form_power(n, treatment_effect) for n in n_range]

                    fig, ax = plt.subplots()
                    ax.plot(n_range, power_curve)
                    ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
                    ax.axvline(n_control, color="red", linestyle="--", label=f"required n ({n_control:.1f})")
                    ax.set_xlabel("Control arm sample size")
                    ax.set_ylabel("Power")
                    ax.legend()
                    st.pyplot(fig)

                    if run_simulation:
                        with st.spinner("Running simulation..."):
                            sim_power = trial.simulate_power(
                                n_per_arm=round(n_control), treatment_effect=treatment_effect, n_sims=n_sims, seed=1,
                            )
                        st.write(f"Simulated power at control n={round(n_control)}: {sim_power:.3f} (target: {power_target})")
                        st.caption(
                            "Simulated power can exceed the closed-form target when the baseline-outcome correlation "
                            "is high. Adjusting for a predictive baseline covariate increases power beyond the "
                            "closed-form assumption."
                        )

    else:  # Non-inferiority
        margin = st.number_input(
            "Non-inferiority margin", value=2.0, min_value=0.01,
            help="Largest acceptable amount by which the treatment mean could be worse than control and still be declared non-inferior.",
        )
        assumed_true_effect = st.number_input(
            "Assumed true treatment effect (mean difference)", value=0.0,
            help="The actual difference assumed to exist between arms for planning purposes. 0 (treatment truly equal to control) is the standard conservative assumption.",
        )
        outcome_sd = st.number_input(
            "Outcome standard deviation", value=10.0, min_value=0.01,
            help="Natural variability in outcome between subjects, independent of treatment.",
        )
        power_target = st.slider(
            "Target power", 0.5, 0.99, 0.8,
            help="Probability of declaring non-inferiority at the calculated sample size, assuming the specified true effect.",
        )
        alpha = st.number_input(
            "Alpha (one-sided significance level)", value=0.025, min_value=0.001, max_value=0.5,
            help="Acceptable false-positive rate for the one-sided non-inferiority test. Regulatory convention commonly uses 0.025 one-sided, considered equivalent rigor to a two-sided 0.05 test.",
        )
        allocation_ratio = st.number_input(
            "Allocation ratio (treatment:control)", value=1.0, min_value=0.1,
            help="Ratio of treatment arm size to control arm size. A value of 1.0 indicates equal allocation. A value of 2.0 indicates twice as many subjects in the treatment arm as the control arm.",
        )

        if st.button("Calculate"):
            trial = ContinuousEndpoint(outcome_sd=outcome_sd, alpha=alpha, allocation_ratio=allocation_ratio)
            try:
                n_control = trial.closed_form_sample_size_ni(margin=margin, treatment_effect=assumed_true_effect, power=power_target)
            except RuntimeError:
                st.error(
                    "The required sample size exceeds 10 million subjects per arm, far beyond any "
                    "realistic trial. This margin is too small relative to the outcome standard "
                    "deviation to be practically achievable. Consider a larger margin or a smaller "
                    "standard deviation."
                )
            else:
                n_treatment = n_control * allocation_ratio
                st.write(f"Control arm sample size: {n_control:.1f}")
                st.write(f"Treatment arm sample size: {n_treatment:.1f}")
                st.write(f"Total sample size: {n_control + n_treatment:.1f}")

                st.subheader("Power curve")
                n_range = np.linspace(5, n_control * 2, 100)
                power_curve = [trial.closed_form_power_ni(n, margin, assumed_true_effect) for n in n_range]

                fig, ax = plt.subplots()
                ax.plot(n_range, power_curve)
                ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
                ax.axvline(n_control, color="red", linestyle="--", label=f"required n ({n_control:.1f})")
                ax.set_xlabel("Control arm sample size")
                ax.set_ylabel("Power")
                ax.legend()
                st.pyplot(fig)
elif endpoint_type == "Binary":
    test_type = st.radio(
        "Test type",
        ["Superiority", "Non-inferiority"],
        help="Superiority tests whether the treatment differs from control. Non-inferiority tests whether the treatment is not worse than control by more than a specified margin.",
    )

    if test_type == "Superiority":
        control_event_rate = st.slider(
            "Control arm event rate", 0.01, 0.99, 0.30,
            help="Proportion of control-arm subjects expected to experience the event.",
        )
        odds_ratio = st.number_input(
            "Treatment odds ratio", value=0.5, min_value=0.01,
            help="Change in the odds of the event under treatment. Values below 1.0 indicate reduced odds. Values above 1.0 indicate increased odds.",
        )
        power_target = st.slider(
            "Target power", 0.5, 0.99, 0.8,
            help="Probability of detecting the treatment effect at the calculated sample size, assuming the effect is real.",
        )
        alpha = st.number_input(
            "Alpha (significance level)", value=0.05, min_value=0.001, max_value=0.5,
            help="Acceptable false-positive rate. Conventional default is 0.05.",
        )
        allocation_ratio = st.number_input(
            "Allocation ratio (treatment:control)", value=1.0, min_value=0.1,
            help="Ratio of treatment arm size to control arm size. A value of 1.0 indicates equal allocation. A value of 2.0 indicates twice as many subjects in the treatment arm as the control arm.",
        )

        run_simulation = st.checkbox(
            "Also validate with simulation-based power",
            help="Generates synthetic trial data and fits a logistic regression model directly to estimate power. Slower than the closed-form formula. Reflects the result of an actual regression-based analysis.",
        )
        if run_simulation:
            n_sims = st.number_input("Number of simulations", value=2000, min_value=100, step=100)

        if st.button("Calculate"):
            if odds_ratio == 1.0:
                st.error("Treatment odds ratio is 1.0, meaning no difference between arms. No finite sample size exists when there is no difference between arms. Enter an odds ratio other than 1.0.")
            else:
                trial = BinaryEndpoint(alpha=alpha, allocation_ratio=allocation_ratio)
                odds_control = control_event_rate / (1 - control_event_rate)
                odds_treatment = odds_control * odds_ratio
                p_treatment = odds_treatment / (1 + odds_treatment)

                try:
                    n_control = trial.closed_form_sample_size(p1=control_event_rate, p2=p_treatment, power=power_target)
                except ValueError:
                    st.error("Could not calculate a sample size for these inputs. Check that the control event rate and odds ratio produce a realistic treatment event rate.")
                else:
                    n_treatment = n_control * allocation_ratio
                    st.write(f"Treatment arm event rate implied by this odds ratio: {p_treatment:.3f}")
                    st.write(f"Control arm sample size: {n_control:.1f}")
                    st.write(f"Treatment arm sample size: {n_treatment:.1f}")
                    st.write(f"Total sample size: {n_control + n_treatment:.1f}")

                    st.subheader("Power curve")
                    n_range = np.linspace(5, n_control * 2, 100)
                    power_curve = [trial.closed_form_power(n, control_event_rate, p_treatment) for n in n_range]

                    fig, ax = plt.subplots()
                    ax.plot(n_range, power_curve)
                    ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
                    ax.axvline(n_control, color="red", linestyle="--", label=f"required n ({n_control:.1f})")
                    ax.set_xlabel("Control arm sample size")
                    ax.set_ylabel("Power")
                    ax.legend()
                    st.pyplot(fig)

                    if run_simulation:
                        with st.spinner("Running simulation..."):
                            sim_power = trial.simulate_power(
                                n_per_arm=round(n_control), control_event_rate=control_event_rate,
                                odds_ratio=odds_ratio, n_sims=n_sims, seed=1,
                            )
                        st.write(f"Simulated power at control n={round(n_control)}: {sim_power:.3f} (target: {power_target})")
                        st.caption(
                            "Simulated power can differ from the closed-form target because the closed-form "
                            "formula assumes a two-proportion test, while the simulation fits logistic "
                            "regression directly. Logistic regression's Wald test can produce power modestly "
                            "below the closed-form target."
                        )

    else:  # Non-inferiority
        st.caption(
            "Assumes lower event rates are better (e.g. an adverse event or failure endpoint). "
            "Treatment is non-inferior if its event rate is not more than the margin higher than control's."
        )

        control_event_rate = st.slider(
            "Control arm event rate", 0.01, 0.99, 0.20,
            help="Proportion of control-arm subjects expected to experience the event.",
        )
        margin = st.slider(
            "Non-inferiority margin (percentage points)", 0.01, 0.50, 0.10,
            help="Largest acceptable amount by which the treatment event rate could exceed the control event rate and still be declared non-inferior.",
        )
        assumed_treatment_rate = st.slider(
            "Assumed true treatment event rate", 0.01, 0.99, 0.20,
            help="The actual treatment event rate assumed for planning purposes. Equal to the control rate (treatment truly equal to control) is the standard conservative assumption.",
        )
        power_target = st.slider(
            "Target power", 0.5, 0.99, 0.8,
            help="Probability of declaring non-inferiority at the calculated sample size, assuming the specified true rates.",
        )
        alpha = st.number_input(
            "Alpha (one-sided significance level)", value=0.025, min_value=0.001, max_value=0.5,
            help="Acceptable false-positive rate for the one-sided non-inferiority test. Regulatory convention commonly uses 0.025 one-sided, considered equivalent rigor to a two-sided 0.05 test.",
        )
        allocation_ratio = st.number_input(
            "Allocation ratio (treatment:control)", value=1.0, min_value=0.1,
            help="Ratio of treatment arm size to control arm size. A value of 1.0 indicates equal allocation. A value of 2.0 indicates twice as many subjects in the treatment arm as the control arm.",
        )

        if st.button("Calculate"):
            trial = BinaryEndpoint(alpha=alpha, allocation_ratio=allocation_ratio)
            try:
                n_control = trial.closed_form_sample_size_ni(
                    margin=margin, p_control=control_event_rate, p_treatment=assumed_treatment_rate, power=power_target,
                )
            except ValueError:
                st.error("Could not calculate a sample size for these inputs. Check that the control rate, margin, and assumed treatment rate are realistic together.")
            else:
                n_treatment = n_control * allocation_ratio
                st.write(f"Control arm sample size: {n_control:.1f}")
                st.write(f"Treatment arm sample size: {n_treatment:.1f}")
                st.write(f"Total sample size: {n_control + n_treatment:.1f}")

                st.subheader("Power curve")
                n_range = np.linspace(5, n_control * 2, 100)
                power_curve = [
                    trial.closed_form_power_ni(n, margin, control_event_rate, assumed_treatment_rate)
                    for n in n_range
                ]

                fig, ax = plt.subplots()
                ax.plot(n_range, power_curve)
                ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
                ax.axvline(n_control, color="red", linestyle="--", label=f"required n ({n_control:.1f})")
                ax.set_xlabel("Control arm sample size")
                ax.set_ylabel("Power")
                ax.legend()
                st.pyplot(fig)
elif endpoint_type == "Survival":
    test_type = st.radio(
        "Test type",
        ["Superiority", "Non-inferiority"],
        help="Superiority tests whether the treatment differs from control. Non-inferiority tests whether the treatment is not worse than control by more than a specified margin.",
    )

    st.caption("Enter all time values in a consistent unit. Months is typical for clinical trials.")

    if test_type == "Superiority":
        control_median_survival = st.number_input(
            "Control arm median survival (months)", value=12.0, min_value=0.01,
            help="Time by which 50% of control-arm subjects are expected to experience the event. All time-based fields below must use the same unit.",
        )
        hazard_ratio = st.number_input(
            "Treatment hazard ratio", value=0.6, min_value=0.01,
            help="Change in the instantaneous risk of the event under treatment. Values below 1.0 indicate reduced risk. Values above 1.0 indicate increased risk.",
        )
        accrual_period = st.number_input(
            "Accrual period (months)", value=6.0, min_value=0.01,
            help="Duration over which subjects are enrolled at a steady rate, before the study closes to new subjects.",
        )
        follow_up_period = st.number_input(
            "Follow-up period (months, after last enrollment)", value=18.0, min_value=0.01,
            help="Duration subjects are followed after the last subject enrolls, before study end and administrative censoring.",
        )
        power_target = st.slider(
            "Target power", 0.5, 0.99, 0.8,
            help="Probability of detecting the treatment effect at the calculated sample size, assuming the effect is real.",
        )
        alpha = st.number_input(
            "Alpha (significance level)", value=0.05, min_value=0.001, max_value=0.5,
            help="Acceptable false-positive rate. Conventional default is 0.05.",
        )
        allocation_ratio = st.number_input(
            "Allocation ratio (treatment:control)", value=1.0, min_value=0.1,
            help="Ratio of treatment arm size to control arm size. A value of 1.0 indicates equal allocation. A value of 2.0 indicates twice as many subjects in the treatment arm as the control arm.",
        )

        run_simulation = st.checkbox(
            "Also validate with simulation-based power",
            help="Generates synthetic trial data and fits a Cox proportional hazards model directly to estimate power. Slower than the closed-form formula. Reflects the result of an actual regression-based analysis.",
        )
        if run_simulation:
            n_sims = st.number_input("Number of simulations", value=300, min_value=50, step=50)
            st.caption("Cox model fits are slower than the other endpoints, fewer simulations by default.")

        if st.button("Calculate"):
            if hazard_ratio == 1.0:
                st.error("Treatment hazard ratio is 1.0, meaning no difference between arms. No finite sample size exists when there is no difference between arms. Enter a hazard ratio other than 1.0.")
            else:
                trial = SurvivalEndpoint(
                    accrual_period=accrual_period, follow_up_period=follow_up_period,
                    alpha=alpha, allocation_ratio=allocation_ratio,
                )
                result = trial.closed_form_sample_size(
                    hazard_ratio=hazard_ratio, control_median_survival=control_median_survival, power=power_target,
                )
                st.write(f"Events needed: {result['events_needed']:.1f}")
                st.write(f"Control arm sample size: {result['n_control']:.1f}")
                st.write(f"Treatment arm sample size: {result['n_treatment']:.1f}")
                st.write(f"Total sample size: {result['n_total']:.1f}")

                st.subheader("Power curve")
                n_control_point = result["n_control"]
                n_range = np.linspace(5, n_control_point * 2, 100)
                power_curve = [trial.closed_form_power(n, hazard_ratio, control_median_survival) for n in n_range]

                fig, ax = plt.subplots()
                ax.plot(n_range, power_curve)
                ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
                ax.axvline(n_control_point, color="red", linestyle="--", label=f"required n ({n_control_point:.1f})")
                ax.set_xlabel("Control arm sample size")
                ax.set_ylabel("Power")
                ax.legend()
                st.pyplot(fig)

                if run_simulation:
                    with st.spinner("Running simulation (this one's slower)..."):
                        sim_power = trial.simulate_power(
                            n_per_arm=round(result["n_control"]), control_median_survival=control_median_survival,
                            hazard_ratio=hazard_ratio, n_sims=n_sims, seed=1,
                        )
                    st.write(f"Simulated power at control n={round(result['n_control'])}: {sim_power:.3f} (target: {power_target})")
                    st.caption(
                        "Simulated power can differ from the closed-form target because the closed-form formula "
                        "(Schoenfeld's formula) assumes a log-rank test, while the simulation fits a Cox "
                        "proportional hazards model directly. The Cox model's Wald test can produce power "
                        "modestly below the closed-form target."
                    )

    else:  # Non-inferiority
        control_median_survival = st.number_input(
            "Control arm median survival (months)", value=12.0, min_value=0.01,
            help="Time by which 50% of control-arm subjects are expected to experience the event. All time-based fields below must use the same unit.",
        )
        margin = st.number_input(
            "Non-inferiority margin (hazard ratio)", value=1.3, min_value=1.01,
            help="Largest hazard ratio (treatment vs. control) still considered non-inferior. Values above 1.0 indicate treatment can have somewhat higher risk and still be acceptable.",
        )
        true_hazard_ratio = st.number_input(
            "Assumed true hazard ratio", value=1.0, min_value=0.01,
            help="The actual hazard ratio assumed for planning purposes. 1.0 (treatment truly equal to control) is the standard conservative assumption.",
        )
        accrual_period = st.number_input(
            "Accrual period (months)", value=6.0, min_value=0.01,
            help="Duration over which subjects are enrolled at a steady rate, before the study closes to new subjects.",
        )
        follow_up_period = st.number_input(
            "Follow-up period (months, after last enrollment)", value=18.0, min_value=0.01,
            help="Duration subjects are followed after the last subject enrolls, before study end and administrative censoring.",
        )
        power_target = st.slider(
            "Target power", 0.5, 0.99, 0.8,
            help="Probability of declaring non-inferiority at the calculated sample size, assuming the specified true hazard ratio.",
        )
        alpha = st.number_input(
            "Alpha (one-sided significance level)", value=0.025, min_value=0.001, max_value=0.5,
            help="Acceptable false-positive rate for the one-sided non-inferiority test. Regulatory convention commonly uses 0.025 one-sided, considered equivalent rigor to a two-sided 0.05 test.",
        )
        allocation_ratio = st.number_input(
            "Allocation ratio (treatment:control)", value=1.0, min_value=0.1,
            help="Ratio of treatment arm size to control arm size. A value of 1.0 indicates equal allocation. A value of 2.0 indicates twice as many subjects in the treatment arm as the control arm.",
        )

        if st.button("Calculate"):
            trial = SurvivalEndpoint(
                accrual_period=accrual_period, follow_up_period=follow_up_period,
                alpha=alpha, allocation_ratio=allocation_ratio,
            )
            result = trial.closed_form_sample_size_ni(
                margin=margin, control_median_survival=control_median_survival,
                true_hazard_ratio=true_hazard_ratio, power=power_target,
            )
            st.write(f"Events needed: {result['events_needed']:.1f}")
            st.write(f"Control arm sample size: {result['n_control']:.1f}")
            st.write(f"Treatment arm sample size: {result['n_treatment']:.1f}")
            st.write(f"Total sample size: {result['n_total']:.1f}")

            st.subheader("Power curve")
            n_control_point = result["n_control"]
            n_range = np.linspace(5, n_control_point * 2, 100)
            power_curve = [
                trial.closed_form_power_ni(n, margin, control_median_survival, true_hazard_ratio)
                for n in n_range
            ]

            fig, ax = plt.subplots()
            ax.plot(n_range, power_curve)
            ax.axhline(power_target, color="gray", linestyle="--", label=f"target power ({power_target})")
            ax.axvline(n_control_point, color="red", linestyle="--", label=f"required n ({n_control_point:.1f})")
            ax.set_xlabel("Control arm sample size")
            ax.set_ylabel("Power")
            ax.legend()
            st.pyplot(fig)
