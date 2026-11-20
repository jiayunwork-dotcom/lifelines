# -*- coding: utf-8 -*-
"""
Synthetic-data experiment for SurvivalCurveCalibrator.

ALL COHORTS BELOW ARE SYNTHETIC, generated in this script with fixed seeds.
No real-world data is used.

Protocol (fixed in advance)
---------------------------
* Three independent synthetic cohorts are drawn: train / calibrate / test.
* The failure-time distribution is known (Weibull AFT):  log T = -log lam(x) + eps / rho
  with Gumbel eps, covariates x1 ~ N(0,1), x2 ~ Bernoulli(0.5), and independent
  exponential censoring.
* Scenario A (miscalibrated base model): the train cohort is drawn with a
  *different baseline* (shifted intercept, different shape) than the
  calibrate/test cohorts. The fitted model keeps a useful risk ranking but its
  absolute probabilities are deliberately off -- the setting described in the
  issue ("sorting still works, reported probabilities are off").
* Scenario B (already calibrated base model): all three cohorts share one
  distribution. Recalibration should be close to the identity map and neither
  help nor hurt much; any difference is finite-sample noise.
* Comparison metric: out-of-sample integrated Brier score (IBS) over the
  calibration time grid with inverse-probability-of-censoring weights, the
  censoring distribution estimated on the *test* cohort (standard evaluation
  practice; the cross-validation machinery estimates it on each fold's
  calibrate rows instead). Lower IBS is better.
* Fixed seeds and sample sizes are used throughout; the script prints both
  scenarios and a k_fold_cross_validation demonstration.

Run with:  python3 experiments/calibration_curve_experiment.py
"""

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter, WeibullAFTFitter, SurvivalCurveCalibrator
from lifelines.utils import k_fold_cross_validation

# ------------------------------------------------------- fixed protocol
SEED_TRAIN, SEED_CAL, SEED_TEST = 20261, 20262, 20263
N_TRAIN, N_CAL, N_TEST = 800, 500, 1000
GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
BETA = np.array([0.6, -0.9])  # covariate effects, shared by all cohorts
CENSOR_RATE = 6.0  # exponential censoring rate


def make_cohort(seed, n, intercept, shape):
    """One synthetic cohort from a Weibull AFT model with known parameters."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, n)
    lam = np.exp(intercept + BETA[0] * x1 + BETA[1] * x2)
    event_time = lam * rng.weibull(shape, n)
    censor_time = rng.exponential(CENSOR_RATE, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "T": np.minimum(event_time, censor_time),
            "E": (event_time <= censor_time).astype(float),
        }
    )


def ipcw_integrated_brier(model, test_df, grid, calibration=None):
    """Time-averaged IPCW Brier score on the test cohort (lower is better).

    The censoring distribution G is estimated on the test cohort itself.
    """
    T = test_df["T"].to_numpy()
    E = test_df["E"].to_numpy()
    # Kaplan-Meier of the censoring distribution on the test cohort
    censor_times = np.sort(T[E == 0])
    uniq = np.unique(censor_times)
    g_values = np.ones(uniq.size)
    surv = 1.0
    for i, t in enumerate(uniq):
        surv *= 1.0 - np.sum(censor_times == t) / np.sum(T >= t)
        g_values[i] = surv

    def G(t):
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(uniq, t, side="right")
        return np.concatenate(([1.0], g_values))[idx]

    if calibration is None:
        s_hat = model.predict_survival_function(test_df, times=grid).to_numpy()
    else:
        s_hat = model.predict_survival_function(test_df, times=grid, calibration=calibration).to_numpy()

    bs = np.zeros(grid.size)
    for j, t in enumerate(grid):
        event_seen = (T <= t) & (E == 1)
        at_risk = T > t
        loss = np.sum(s_hat[j, event_seen] ** 2 / G(T[event_seen]))
        loss += np.sum((1.0 - s_hat[j, at_risk]) ** 2 / G(t))
        bs[j] = loss / T.size
    return float(np.trapezoid(bs, grid) / (grid[-1] - grid[0]))


def map_deviation_from_identity(calibrator, model, df, grid):
    """Mean |phi(u) - u| over the pooled log-cumulative-hazards: 0 == identity."""
    raw = model.predict_survival_function(df, times=grid).to_numpy()
    u = np.log(np.clip(-np.log(np.clip(raw, 1e-300, 1.0)), 1e-300, None))
    phi = calibrator._phi(u, calibrator._knots, calibrator._intercept, np.exp(calibrator._log_slopes))
    return float(np.mean(np.abs(phi - u)))


def run_scenario(name, intercept_train, shape_train, intercept_test, shape_test):
    print("=" * 78)
    print(name)
    print("=" * 78)
    train = make_cohort(SEED_TRAIN, N_TRAIN, intercept_train, shape_train)
    cal = make_cohort(SEED_CAL, N_CAL, intercept_test, shape_test)
    test = make_cohort(SEED_TEST, N_TEST, intercept_test, shape_test)
    print(
        "cohort sizes: train=%d calibrate=%d test=%d | event rates: %.2f / %.2f / %.2f"
        % (len(train), len(cal), len(test), train.E.mean(), cal.E.mean(), test.E.mean())
    )

    results = {}
    for label, model in [("CoxPHFitter", CoxPHFitter()), ("WeibullAFTFitter", WeibullAFTFitter())]:
        model.fit(train, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)

        ibs_raw = ipcw_integrated_brier(model, test, GRID)
        ibs_cal = ipcw_integrated_brier(model, test, GRID, calibration=calibrator)
        deviation = map_deviation_from_identity(calibrator, model, cal, GRID)

        # sanity: calibrated curves are valid survival curves
        curves = model.predict_survival_function(test, calibration=calibrator).to_numpy()
        assert np.all(curves >= 0) and np.all(curves <= 1)
        assert np.all(np.diff(curves, axis=0) <= 1e-12)
        # sanity: not a constant output
        assert curves.std() > 0

        results[label] = (ibs_raw, ibs_cal)
        print(
            "%-16s  IBS raw = %.4f   IBS calibrated = %.4f   (delta = %+.4f)   map deviation from identity = %.3f"
            % (label, ibs_raw, ibs_cal, ibs_cal - ibs_raw, deviation)
        )
    return train, cal, test, results


def main():
    # ---------------- Scenario A: base model deliberately miscalibrated
    train_a, cal_a, test_a, _ = run_scenario(
        "Scenario A: baseline shift between train and calibrate/test (miscalibrated)",
        intercept_train=2.6,
        shape_train=1.2,
        intercept_test=1.8,
        shape_test=1.6,
    )

    # ---------------- Scenario B: base model already calibrated
    run_scenario(
        "Scenario B: one shared distribution (already calibrated)",
        intercept_train=1.8,
        shape_train=1.6,
        intercept_test=1.8,
        shape_test=1.6,
    )

    # ---------------- k_fold_cross_validation evaluates the calibration too
    print("=" * 78)
    print("k_fold_cross_validation with calibration_cv (scenario A cohorts, one fixed split)")
    print("=" * 78)
    pooled = pd.concat([train_a, cal_a, test_a], ignore_index=True)
    n_train, n_cal = len(train_a), len(cal_a)
    folds = iter(
        [
            (
                np.arange(0, n_train),
                np.arange(n_train, n_train + n_cal),
                np.arange(n_train + n_cal, n_train + n_cal + len(test_a)),
            )
        ]
    )
    scores = k_fold_cross_validation(
        [CoxPHFitter(), WeibullAFTFitter()],
        pooled,
        "T",
        "E",
        scoring_method="integrated_brier_score",
        calibration_cv=folds,
        calibrator_factory=SurvivalCurveCalibrator,
        calibration_times=GRID,
    )
    for label, model_scores in zip(["CoxPHFitter", "WeibullAFTFitter"], scores):
        print("%-16s  calibrated CV scores (negative IBS, higher is better): %s" % (label, model_scores))


if __name__ == "__main__":
    main()
