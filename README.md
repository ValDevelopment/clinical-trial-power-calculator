---
noteId: "4a00be10817711f1aed8cfa6badee3f5"
tags: []

---

# Clinical Trial Power Calculator

A sample size and power calculator for two-arm clinical trials, covering continuous, binary, and time-to-event endpoints. Supports both closed-form formulas and simulation-based power estimated by fitting the actual regression model (ANCOVA, logistic regression, Cox proportional hazards) to synthetic trial data, rather than relying on textbook approximations alone.

## Objective

Sample size planning is a design-stage problem: before a trial exists, a statistician must determine how many subjects are required to detect an effect of a given size at a specified level of confidence. This tool addresses that question in two ways for each endpoint type: a closed-form formula for rapid estimation, and a simulation that generates synthetic subject-level data under the specified assumptions and fits the actual analysis model to it, producing an empirical power estimate that reflects what the planned analysis would find in practice.

This project is deliberately prospective rather than retrospective. The [CDISC safety analysis](../cdisc-safety-portfolio) and [FAERS signal detection](../faers-signal-detection) projects both analyze data that already exists. This tool supports the decision made before a trial begins, addressing a different phase of a trial's life cycle than either of those projects.

## Approach

Unlike a typical analysis project, no external dataset is used. The underlying data is synthetic, generated from user-specified design assumptions (effect size, variance, dropout, accrual, and related parameters). This is consistent with how established power analysis software operates (G*Power, PASS, and R's `pwr` package all follow the same approach), and reflects a recognized methodology often referred to as clinical trial simulation. FDA guidance on adaptive and Bayesian trial designs explicitly calls for simulation-based justification of operating characteristics for designs that lack closed-form solutions, which is the role the simulation module serves here.

## Methodology

Each endpoint type has a data generating process (DGP), a closed-form power/sample-size formula, and a simulation-based power estimator, implemented as a class in `power_calc/`.

**Continuous endpoint** (`ContinuousEndpoint`)
- DGP: outcome depends on a baseline covariate (configurable correlation) plus a baseline-adjusted treatment effect
- Closed-form: two-sample t-test power via the noncentral t-distribution (`scipy.stats.nct`), with sample size determined by root-finding (`scipy.optimize.brentq`)
- Simulation: fits ANCOVA (OLS regression of outcome on arm and baseline) to each replicate via `statsmodels`; power is defined as the proportion of replicates in which the arm coefficient is statistically significant

**Binary endpoint** (`BinaryEndpoint`)
- DGP: logistic model, parameterized by a control event rate and a treatment odds ratio
- Closed-form: two-proportion test via the arcsine (Cohen's h) method, consistent with the approach used in R's `pwr.2p.test`
- Simulation: fits logistic regression (`statsmodels.Logit`) to each replicate

**Time-to-event endpoint** (`SurvivalEndpoint`)
- DGP: constant-hazard (exponential) proportional hazards model, incorporating uniform accrual, administrative censoring at study end, and an optional independent dropout hazard for loss to follow-up
- Closed-form: Schoenfeld's formula for the number of events required, converted to a subject count using a derived closed-form expected event probability under uniform accrual and administrative censoring
- Simulation: fits Cox proportional hazards regression (`statsmodels.PHReg`) to each replicate

## Validation

Each closed-form formula was verified against an independent source (a published benchmark, an existing implementation in `statsmodels`, or a direct simulation of the underlying test) prior to use, and each simulation-based power estimate was compared against its corresponding closed-form value. This process produced two consistent findings, rather than artifacts of the implementation:

| Endpoint | n (per arm) | Closed-form target | Simulated power (regression) | Finding |
|---|---|---|---|---|
| Continuous | 64 | 0.80 (t-test) | 0.8025 (ANCOVA, baseline correlation = 0) | Matches the closed-form result when the baseline covariate has no adjustment value |
| Continuous | 64 | 0.80 (t-test) | 0.9715 (ANCOVA, baseline correlation = 0.7) | ANCOVA yields a meaningful power advantage over an unadjusted t-test when the baseline covariate is predictive of outcome |
| Binary | 182 | 0.80 (two-proportion z) | 0.7845 (logistic regression) | The logistic regression Wald test produces power somewhat below a closed-form target based on a simpler test |
| Survival | 99 | 0.80 (log-rank / Schoenfeld) | 0.7740 (Cox proportional hazards) | The same Wald-test gap is observed, and persists independent of covariate adjustment (an arm-only Cox model produces an equivalent result) |

The gap observed for the binary and survival endpoints does not indicate an implementation error. It reflects an expected and well-documented difference between the test a closed-form formula assumes (a two-proportion test, a log-rank test) and the test an actual regression-based analysis performs (a Wald test derived from a fitted model). A closed-form sample size serves as a rapid design-stage estimate; where the planned analysis involves covariate adjustment through regression, the simulation module determines whether that estimate remains adequate.

One additional implementation-level finding is documented here: `scipy.stats.nct.cdf`, used in the continuous endpoint's closed-form power calculation, returns `NaN` for certain large combinations of degrees of freedom and noncentrality. This is a numerical instability internal to `scipy`, not a defect in the underlying formula. `closed_form_sample_size` avoids this condition by expanding its root-finding bracket geometrically from a small starting point rather than searching a fixed, arbitrarily wide range, which keeps the search away from the unstable region. This approach was verified against 500 randomized parameter combinations with zero failures.

## Worked Example

`notebooks/04_worked_example.ipynb` applies the survival endpoint to a finding from the CDISC safety analysis project: a hazard ratio of 5.03 for dose-dependent adverse event risk. A confirmatory study powered around this observed effect requires approximately 15 total subjects, reflecting the magnitude of the effect. A more conservative planning assumption (HR = 1.5) requires approximately 293 subjects, nearly twenty times as many. This comparison illustrates the notebook's central point: required sample size for a confirmatory trial depends substantially on the level of confidence placed in a pilot study's effect size estimate, a judgment this tool is intended to support rather than replace.

## Repository Structure

```
clinical-trial-power-calculator/
├── power_calc/
│   ├── __init__.py
│   ├── continuous.py
│   ├── binary.py
│   └── survival.py
├── notebooks/
│   ├── 01_data_generating_process.ipynb
│   ├── 02_closed_form_power.ipynb
│   ├── 03_simulation_power.ipynb
│   └── 04_worked_example.ipynb
├── tests/
│   ├── test_continuous.py
│   ├── test_binary.py
│   └── test_survival.py
├── app.py
├── pyproject.toml
└── README.md
```

## Usage

```bash
pip install -e .
pytest tests/            # confirm the validated benchmarks still hold
streamlit run app.py      # interactive calculator
```

```python
from power_calc.continuous import ContinuousEndpoint

trial = ContinuousEndpoint(outcome_sd=10.0, baseline_outcome_corr=0.5)
n = trial.closed_form_sample_size(treatment_effect=5.0, power=0.8)
sim_power = trial.simulate_power(n_per_arm=round(n), treatment_effect=5.0, n_sims=2000, seed=1)
```

## Streamlit Application

`app.py` provides an interactive interface for all three endpoint types: closed-form sample size calculation, an optional simulation-based validation step (with a reproducible seed), and a power curve plot illustrating how power scales with sample size under the specified design assumptions.

## Limitations

- Dropout is modeled as missing completely at random (MCAR) throughout. In practice, dropout often depends on baseline covariates or the outcome itself (MAR/MNAR), which this tool does not model.
- The survival endpoint assumes a constant hazard (exponential distribution). Real hazard functions are frequently non-constant; a Weibull or piecewise-constant option would be a natural extension.
- Simulation-based power relies on complete-case analysis (excluding missing outcomes), consistent with the MCAR assumption above but not representative of how a trial with informative missingness would typically be analyzed (for example, via MMRM or multiple imputation).
- Closed-form formulas target standard tests (t-test, two-proportion z, log-rank). As shown in Validation, an actual regression-based analysis can produce modestly different power, in some cases higher (covariate adjustment) and in others lower (Wald test finite-sample behavior). The simulation module is designed specifically to identify this gap.
- Non-inferiority margins, unequal allocation beyond a fixed ratio, and interim or group sequential designs are not yet supported.
- Cox proportional hazards simulation is notably slower than the other two endpoints (approximately 70ms per replicate); its default simulation count is set lower accordingly.

## References

- Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. *Biometrics*. 1983;39(2):499-503.
- Cohen J. *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. 1988. (arcsine transformation for proportions)
- FDA guidance on adaptive and Bayesian clinical trial designs, regarding the use of simulation to characterize operating characteristics for designs without closed-form solutions.
