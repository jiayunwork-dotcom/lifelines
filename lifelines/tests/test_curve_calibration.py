# -*- coding: utf-8 -*-
"""
Tests for SurvivalCurveCalibrator, the ``calibration=`` prediction path, the
``calibration`` option of survival_probability_calibration, and
calibration-aware k_fold_cross_validation.

All cohorts in this file are SYNTHETIC, generated in-code with fixed seeds.
"""
import numpy as np
import pandas as pd
import pytest

from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter, SurvivalCurveCalibrator
from lifelines.calibration import survival_probability_calibration
from lifelines.utils import k_fold_cross_validation, CensoringType

RTOL, ATOL = 1e-7, 1e-9

# ------------------------------------------------------------------ helpers
# Fixed protocol for the learning experiments: seeds, sizes, metric.
SEED_TRAIN, SEED_CAL, SEED_TEST = 20261, 20262, 20263
N_TRAIN, N_CAL, N_TEST = 800, 500, 1000
GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
BETA = np.array([0.6, -0.9])
CENSOR_RATE = 6.0


def make_cohort(seed, n, intercept=1.8, shape=1.6, censor_rate=CENSOR_RATE):
    """Synthetic cohort from a Weibull AFT model with known parameters."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, n)
    lam = np.exp(intercept + BETA[0] * x1 + BETA[1] * x2)
    event_time = lam * rng.weibull(shape, n)
    censor_time = rng.exponential(censor_rate, n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "T": np.minimum(event_time, censor_time),
            "E": (event_time <= censor_time).astype(float),
        }
    )


def ipcw_ibs(model, test_df, grid, calibration=None):
    """Independent reference IPCW integrated Brier score (lower is better).

    G is estimated on ``test_df`` by a hand-rolled Kaplan-Meier of censoring.
    """
    T = test_df["T"].to_numpy()
    E = test_df["E"].to_numpy()
    censor_times = np.sort(T[E == 0])
    uniq = np.unique(censor_times)
    g_at_uniq = []
    surv = 1.0
    for t in uniq:
        surv *= 1.0 - np.sum(censor_times == t) / np.sum(T >= t)
        g_at_uniq.append(surv)
    g_at_uniq = np.asarray(g_at_uniq)

    def G(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.ones(t.size)
        for i, ti in enumerate(t):
            jumped = uniq <= ti
            if jumped.any():
                out[i] = g_at_uniq[jumped][-1]
        return out

    s_hat = np.asarray(model.predict_survival_function(test_df, times=grid, calibration=calibration), dtype=float)
    bs = np.zeros(len(grid))
    for j, t in enumerate(grid):
        for i in range(T.size):
                if T[i] <= t and E[i] == 1:
                    bs[j] += s_hat[j, i] ** 2 / G(T[i])[0]
                elif T[i] > t:
                    bs[j] += (1.0 - s_hat[j, i]) ** 2 / G(t)[0]
        bs[j] /= T.size
    return float(np.trapezoid(bs, grid) / (grid[-1] - grid[0]))


@pytest.fixture(scope="module")
def miscalibrated():
    """Base model fitted on a shifted baseline: ranking useful, probabilities off."""
    train = make_cohort(SEED_TRAIN, N_TRAIN, intercept=2.6, shape=1.2)
    cal = make_cohort(SEED_CAL, N_CAL)
    test = make_cohort(SEED_TEST, N_TEST)
    models = {}
    calibrators = {}
    for key, cls in [("cph", CoxPHFitter), ("aft", WeibullAFTFitter)]:
        model = cls().fit(train, "T", "E")
        models[key] = model
        calibrators[key] = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
    return models, calibrators, cal, test


@pytest.fixture(scope="module")
def well_calibrated():
    """Base model and calibration cohort from one shared distribution."""
    train = make_cohort(SEED_TRAIN, N_TRAIN)
    cal = make_cohort(SEED_CAL, N_CAL)
    test = make_cohort(SEED_TEST, N_TEST)
    model = CoxPHFitter().fit(train, "T", "E")
    calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
    return model, calibrator, test


# ------------------------------------------------------- learning behaviour
class TestLearningGeneralizes:
    def test_miscalibrated_cox_improves_out_of_sample_ibs(self, miscalibrated):
        models, calibrators, cal, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        ibs_raw = ipcw_ibs(model, test, GRID)
        ibs_cal = ipcw_ibs(model, test, GRID, calibration=calibrator)
        assert ibs_cal < ibs_raw - 0.02

    def test_miscalibrated_aft_improves_out_of_sample_ibs(self, miscalibrated):
        models, calibrators, cal, test = miscalibrated
        model, calibrator = models["aft"], calibrators["aft"]
        ibs_raw = ipcw_ibs(model, test, GRID)
        ibs_cal = ipcw_ibs(model, test, GRID, calibration=calibrator)
        assert ibs_cal < ibs_raw - 0.02

    def test_learned_map_is_neither_identity_nor_constant(self, miscalibrated):
        models, calibrators, cal, test = miscalibrated
        for key in ("cph", "aft"):
            model, calibrator = models[key], calibrators[key]
            raw = model.predict_survival_function(cal, times=GRID).to_numpy()
            u = np.log(np.clip(-np.log(np.clip(raw, 1e-300, 1.0)), 1e-300, None))
            phi = calibrator._phi(u, calibrator._knots, calibrator._intercept, np.exp(calibrator._log_slopes))
            # not the identity map
            assert np.mean(np.abs(phi - u)) > 0.1
            out = model.predict_survival_function(test, calibration=calibrator).to_numpy()
            # not a constant output: varies across subjects and across time
            assert out.std(axis=1).mean() > 0.01
            assert out.std(axis=0).mean() > 0.01

    def test_already_calibrated_model_is_left_alone(self, well_calibrated):
        model, calibrator, test = well_calibrated
        ibs_raw = ipcw_ibs(model, test, GRID)
        ibs_cal = ipcw_ibs(model, test, GRID, calibration=calibrator)
        # finite-sample wobble only: recalibrating an already calibrated model
        # must not materially change out-of-sample performance
        assert abs(ibs_cal - ibs_raw) < 0.005


# --------------------------------------------------------- curve constraints
class TestCurveConstraints:
    def test_output_is_a_valid_survival_curve(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        for key in ("cph", "aft"):
            out = models[key].predict_survival_function(test, calibration=calibrators[key]).to_numpy()
            assert np.all(out >= 0.0) and np.all(out <= 1.0)
            assert np.all(np.diff(out, axis=0) <= 1e-12)  # non-increasing in time

    def test_same_time_risk_order_is_never_reversed(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        for key in ("cph", "aft"):
            model, calibrator = models[key], calibrators[key]
            raw = model.predict_survival_function(test, times=GRID).to_numpy()
            cal = model.predict_survival_function(test, times=GRID, calibration=calibrator).to_numpy()
            for j in range(raw.shape[0]):
                order = np.argsort(raw[j], kind="stable")
                calibrated_sorted = cal[j][order]
                # sorting by raw survival must sort calibrated survival (no reversals)
                assert np.all(np.diff(calibrated_sorted) >= -1e-12)

    def test_identical_raw_predictions_get_identical_calibration(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        doubled = pd.concat([test.iloc[[0]], test.iloc[[0]], test.iloc[[1]]])
        out = model.predict_survival_function(doubled, calibration=calibrator).to_numpy()
        np.testing.assert_allclose(out[:, 0], out[:, 1], rtol=RTOL, atol=ATOL)
        assert not np.allclose(out[:, 0], out[:, 2])

    def test_query_grid_density_does_not_change_common_points(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        coarse = model.predict_survival_function(test, times=[2.0, 4.0], calibration=calibrator)
        fine = model.predict_survival_function(
            test, times=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], calibration=calibrator
        )
        np.testing.assert_allclose(coarse.loc[2.0].to_numpy(), fine.loc[2.0].to_numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(coarse.loc[4.0].to_numpy(), fine.loc[4.0].to_numpy(), rtol=RTOL, atol=ATOL)

    def test_shuffled_and_repeated_query_times_keep_positions(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        queried = model.predict_survival_function(test, times=[4.0, 2.0, 4.0, 3.0], calibration=calibrator)
        assert list(queried.index) == [4.0, 2.0, 4.0, 3.0]
        reference = model.predict_survival_function(test, times=[2.0, 3.0, 4.0], calibration=calibrator)
        np.testing.assert_allclose(queried.iloc[0].to_numpy(), reference.loc[4.0].to_numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(queried.iloc[1].to_numpy(), reference.loc[2.0].to_numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(queried.iloc[2].to_numpy(), reference.loc[4.0].to_numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(queried.iloc[3].to_numpy(), reference.loc[3.0].to_numpy(), rtol=RTOL, atol=ATOL)

    def test_duplicate_row_labels_are_output_by_position(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        df = test.iloc[[0, 0, 5]].copy()
        df.index = ["a", "a", "b"]
        with pytest.warns(UserWarning):
            out = model.predict_survival_function(df, calibration=calibrator)
        assert out.shape == (len(GRID), 3)
        # column i is the calibrated curve of row i, regardless of labels
        singles = [model.predict_survival_function(df.iloc[[i]], calibration=calibrator).to_numpy() for i in range(3)]
        for i in range(3):
            np.testing.assert_allclose(out.to_numpy()[:, i], singles[i][:, 0], rtol=RTOL, atol=ATOL)

    def test_default_times_are_the_calibration_grid(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        out = models["cph"].predict_survival_function(test, calibration=calibrators["cph"])
        np.testing.assert_array_equal(out.index.to_numpy(), GRID)

    def test_prediction_does_not_read_outcomes_of_new_subjects(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        covariates_only = test.drop(columns=["T", "E"])
        with_outcomes = test.copy()
        with_outcomes["T"] = 999.0
        with_outcomes["E"] = 1
        a = model.predict_survival_function(covariates_only, calibration=calibrator).to_numpy()
        b = model.predict_survival_function(with_outcomes, calibration=calibrator).to_numpy()
        np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------- input checks
class TestInputValidation:
    def test_query_outside_domain_rejected(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        for bad in ([0.5, 2.0], [7.0], [-1.0, 3.0], [0.0, 3.0], [np.nan, 3.0], [np.inf, 3.0]):
            with pytest.raises(ValueError):
                model.predict_survival_function(test, times=bad, calibration=calibrator)

    def test_conditional_after_rejected_with_calibration(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        with pytest.raises(ValueError):
            models["cph"].predict_survival_function(
                test, calibration=calibrators["cph"], conditional_after=np.ones(len(test))
            )
        with pytest.raises(ValueError):
            models["aft"].predict_survival_function(
                test, calibration=calibrators["aft"], conditional_after=np.ones(len(test))
            )

    def test_non_calibrator_rejected(self, miscalibrated):
        models, _, _, test = miscalibrated
        with pytest.raises(ValueError):
            models["cph"].predict_survival_function(test, calibration=object())

    def test_unfitted_calibrator_rejected(self, miscalibrated):
        models, _, _, test = miscalibrated
        with pytest.raises(ValueError):
            models["cph"].predict_survival_function(test, calibration=SurvivalCurveCalibrator())

    def test_fit_rejects_bad_times(self, miscalibrated):
        models, _, cal, _ = miscalibrated
        model = models["cph"]
        for bad in ([1.0, 2.0], [1.0, 1.0, 2.0], [3.0, 2.0, 1.0], [-1.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, np.nan, 2.0], [1.0, np.inf, 2.0]):
            with pytest.raises(ValueError):
                SurvivalCurveCalibrator().fit(model, cal, times=bad)

    def test_fit_rejects_malformed_data(self, miscalibrated):
        models, _, cal, _ = miscalibrated
        model = models["cph"]
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, cal.iloc[0:0], times=GRID)  # empty
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, cal.drop(columns=["T"]), times=GRID)  # missing duration
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, cal.drop(columns=["E"]), times=GRID)  # missing event
        bad = cal.copy()
        bad.loc[bad.index[0], "T"] = np.inf
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, bad, times=GRID)  # non-finite duration
        bad = cal.copy()
        bad.loc[bad.index[0], "T"] = -1.0
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, bad, times=GRID)  # negative duration
        bad = cal.copy()
        bad.loc[bad.index[0], "E"] = 2
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, bad, times=GRID)  # event not 0/1
        bad = cal.copy()
        bad.loc[bad.index[0], "x1"] = np.nan
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, bad, times=GRID)  # non-finite covariate
        bad = cal.copy()
        bad["x1"] = "a"
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(model, bad, times=GRID)  # non-numeric covariate

    def test_fit_rejects_unsupported_models(self, miscalibrated):
        _, _, cal, _ = miscalibrated
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(CoxPHFitter(), cal, times=GRID)  # not fitted
        kmf = KaplanMeierFitter().fit(cal["T"], cal["E"])
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(kmf, cal, times=GRID)  # univariate model

    def test_fit_rejects_weights_strata_and_entry(self):
        df = make_cohort(SEED_TRAIN, 300)
        cal = make_cohort(SEED_CAL, 200)
        rng = np.random.default_rng(7)

        df_w = df.copy()
        df_w["w"] = 1.0
        cph_w = CoxPHFitter().fit(df_w, "T", "E", weights_col="w")
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(cph_w, cal, times=GRID)

        df_s = df.copy()
        df_s["s"] = rng.binomial(1, 0.5, len(df_s))  # a stratum independent of the covariates
        cph_s = CoxPHFitter().fit(df_s, "T", "E", strata=["s"])
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(cph_s, cal, times=GRID)

        df_e = df.copy()
        df_e["entry"] = 0.0
        aft_e = WeibullAFTFitter().fit(df_e, "T", "E", entry_col="entry")
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator().fit(aft_e, cal, times=GRID)

    def test_no_events_is_a_runtime_error(self, miscalibrated):
        models, _, cal, _ = miscalibrated
        no_events = cal.copy()
        no_events["E"] = 0.0
        with pytest.raises(RuntimeError):
            SurvivalCurveCalibrator().fit(models["cph"], no_events, times=GRID)

    def test_failed_fit_leaves_calibrator_reusable(self, miscalibrated):
        models, _, cal, _ = miscalibrated
        calibrator = SurvivalCurveCalibrator()
        no_events = cal.copy()
        no_events["E"] = 0.0
        with pytest.raises(RuntimeError):
            calibrator.fit(models["cph"], no_events, times=GRID)
        calibrator.fit(models["cph"], cal, times=GRID)  # a legal call afterwards works
        out = models["cph"].predict_survival_function(cal, calibration=calibrator)
        assert out.shape == (len(GRID), len(cal))

    def test_failed_predict_leaves_calibrator_usable(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        with pytest.raises(ValueError):
            model.predict_survival_function(test, times=[100.0], calibration=calibrator)
        out = model.predict_survival_function(test, calibration=calibrator)
        assert out.shape == (len(GRID), len(test))


# --------------------------------------------------------------- model bond
class TestModelBinding:
    def test_other_model_rejected(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        with pytest.raises(ValueError):
            models["aft"].predict_survival_function(test, calibration=calibrators["cph"])

    def test_refit_model_rejected(self):
        # fresh models so the shared module fixtures are not mutated
        train = make_cohort(SEED_TRAIN, N_TRAIN, intercept=2.6, shape=1.2)
        cal = make_cohort(SEED_CAL, N_CAL)
        test = make_cohort(SEED_TEST, N_TEST)
        for cls in (CoxPHFitter, WeibullAFTFitter):
            model = cls().fit(train, "T", "E")
            calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
            model.fit(train, "T", "E")  # refit, even on the same data
            with pytest.raises(ValueError):
                model.predict_survival_function(test, calibration=calibrator)

    def test_fitting_does_not_modify_model_or_inputs(self, miscalibrated):
        models, _, cal, _ = miscalibrated
        model = models["cph"]
        params_before = model.params_.copy()
        cal_copy = cal.copy(deep=True)
        times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=times)
        pd.testing.assert_series_equal(model.params_, params_before)
        pd.testing.assert_frame_equal(cal, cal_copy)
        # mutating the input containers afterwards must not change the learned map
        times[0] = 999.0
        cal.iloc[0, cal.columns.get_loc("T")] = 12345.0
        np.testing.assert_array_equal(calibrator.times_, GRID)
        out = model.predict_survival_function(cal_copy, calibration=calibrator)
        assert out.shape == (len(GRID), len(cal_copy))


# ---------------------------------------------------------------- diagnostics
class TestDiagnostics:
    def test_returns_triple_and_does_not_change_predictions(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        state_before = (calibrator._knots.copy(), calibrator._intercept, calibrator._log_slopes.copy())
        before = model.predict_survival_function(test, times=[3.0], calibration=calibrator)
        ax, ici, e50 = survival_probability_calibration(model, test, 3.0, calibration=calibrator)
        after = model.predict_survival_function(test, times=[3.0], calibration=calibrator)
        assert np.isfinite(ici) and np.isfinite(e50)
        np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())
        # the diagnostic must not refit or mutate the calibrator
        np.testing.assert_array_equal(calibrator._knots, state_before[0])
        assert calibrator._intercept == state_before[1]
        np.testing.assert_array_equal(calibrator._log_slopes, state_before[2])

    def test_diagnostic_uses_the_same_probabilities_as_predict(self, miscalibrated, monkeypatch):
        models, calibrators, _, test = miscalibrated
        model, calibrator = models["cph"], calibrators["cph"]
        seen = {}

        from lifelines import CRCSplineFitter

        original = CRCSplineFitter.fit_right_censoring

        def spy(self, df, *args, **kwargs):
            seen["df"] = df.copy()
            return original(self, df, *args, **kwargs)

        monkeypatch.setattr(CRCSplineFitter, "fit_right_censoring", spy)
        survival_probability_calibration(model, test, 3.0, calibration=calibrator)
        expected = np.log(
            -np.log(model.predict_survival_function(test, times=[3.0], calibration=calibrator).T.squeeze())
        )
        np.testing.assert_allclose(seen["df"]["ccl_at_3"].to_numpy(), expected.to_numpy(), rtol=RTOL, atol=ATOL)

    def test_legacy_call_still_returns_triple(self, miscalibrated):
        models, _, _, test = miscalibrated
        result = survival_probability_calibration(models["cph"], test, 3.0)
        assert len(result) == 3


# ------------------------------------------------- hand-computed IBS checks
class FixedCurveFitter(CoxPHFitter):
    """
    A CoxPHFitter-shaped test double whose predicted survival curves are the
    closed form S_i(t) = exp(-rate_i * t). Ignores any calibrator, so the
    probabilities scored by k_fold_cross_validation are known exactly.
    """

    def __init__(self, rates):
        super().__init__()
        self._rates = np.asarray(rates, dtype=float)
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, df, duration_col, event_col=None, **kwargs):
        self.fit_calls.append(list(df.index))
        self.duration_col = duration_col
        self.event_col = event_col
        self.weights_col = None
        self.entry_col = None
        self.strata = None
        self.params_ = pd.Series([0.0], index=["x"])
        self._time_fit_was_called = "test-double"
        self._censoring_type = CensoringType.RIGHT
        self._model = self  # what CoxPHFitter's facade would delegate to
        return self

    def predict_survival_function(self, X, times=None, conditional_after=None, calibration=None):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        values = np.exp(-np.outer(times, self._rates[:n]))
        return pd.DataFrame(values, index=times, columns=list(range(n)))


def hand_ibs(surv, T, E, grid, censor_T, censor_E):
    """Hand-computed negative time-averaged IPCW Brier score.

    Written directly from the definition, without any of the new library code.
    Convention: G jumps at censoring times and a jump at time s is included in
    G(s); an event at t enters for grid times >= t; a subject censored at t
    still counts as alive for grid times < t only.
    """
    # Kaplan-Meier of censoring by hand
    censor_events = sorted(set(censor_T[censor_E == 0]))
    g_times, g_vals = [], []
    g = 1.0
    for s in censor_events:
        at_risk = np.sum(censor_T >= s)
        d = np.sum((censor_T == s) & (censor_E == 0))
        g *= 1.0 - d / at_risk
        g_times.append(s)
        g_vals.append(g)

    def G(t):
        val = 1.0
        for s, v in zip(g_times, g_vals):
            if s <= t:
                val = v
        return val

    n = len(T)
    bs = []
    for t in grid:
        total = 0.0
        for i in range(n):
            if T[i] <= t and E[i] == 1:
                total += surv(t, i) ** 2 / G(T[i])
            elif T[i] > t:
                total += (1.0 - surv(t, i)) ** 2 / G(t)
        bs.append(total / n)
    ibs = 0.0
    for j in range(len(grid) - 1):
        ibs += 0.5 * (bs[j] + bs[j + 1]) * (grid[j + 1] - grid[j])
    return -ibs / (grid[-1] - grid[0])


class TestHandComputedIBS:
    @staticmethod
    def build_df(rows):
        return pd.DataFrame(rows, columns=["x", "T", "E"])

    def run_cv(self, df, rates, folds, grid):
        fitter = FixedCurveFitter(rates)
        scores = k_fold_cross_validation(
            fitter,
            df,
            "T",
            "E",
            scoring_method="integrated_brier_score",
            calibration_cv=iter(folds),
            calibrator_factory=SurvivalCurveCalibrator,
            calibration_times=grid,
        )
        return scores

    def test_uncensored_scoring_against_hand_computation(self):
        # train rows are dummies (the double does not learn); calibrate rows
        # are events only -> G == 1; test rows are events only
        train_rows = [[0.0, 1.0, 1], [0.0, 2.0, 1], [0.0, 3.0, 1]]
        cal_rows = [[0.0, 1.0, 1], [0.0, 2.0, 1], [0.0, 3.0, 1], [0.0, 4.0, 1], [0.0, 5.0, 1]]
        test_rows = [[0.0, 1.5, 1], [0.0, 2.5, 1], [0.0, 4.5, 1]]
        df = self.build_df(train_rows + cal_rows + test_rows)
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rates = np.array([0.05, 0.2, 0.5, 0.3, 0.4])
        folds = [(np.arange(0, 3), np.arange(3, 8), np.arange(8, 11))]
        scores = self.run_cv(df, rates, folds, grid)

        surv = lambda t, i: np.exp(-rates[i] * t)
        T = np.array([1.5, 2.5, 4.5])
        E = np.array([1, 1, 1])
        expected = hand_ibs(surv, T, E, grid, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1, 1, 1, 1, 1]))
        np.testing.assert_allclose(scores, [expected], rtol=RTOL, atol=ATOL)

    def test_right_censored_scoring_and_same_day_convention(self):
        train_rows = [[0.0, 1.0, 1], [0.0, 2.0, 1], [0.0, 3.0, 1]]
        # calibrate rows carry censoring at t=2 and t=4 -> G jumps there
        cal_rows = [
            [0.0, 1.0, 1],
            [0.0, 2.0, 0],  # censored at 2
            [0.0, 3.0, 1],
            [0.0, 4.0, 0],  # censored at 4
            [0.0, 6.0, 1],
        ]
        test_rows = [
            [0.0, 2.0, 1],  # event the same day as a calibrate censoring
            [0.0, 2.0, 0],  # censored the same day: alive for grid t < 2 only
            [0.0, 4.5, 1],
            [0.0, 6.0, 0],
        ]
        df = self.build_df(train_rows + cal_rows + test_rows)
        grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rates = np.array([0.1, 0.3, 0.6, 0.9, 0.45])
        folds = [(np.arange(0, 3), np.arange(3, 8), np.arange(8, 12))]
        scores = self.run_cv(df, rates, folds, grid)

        surv = lambda t, i: np.exp(-rates[i] * t)
        T = np.array([2.0, 2.0, 4.5, 6.0])
        E = np.array([1, 0, 1, 0])
        cal_T = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
        cal_E = np.array([1, 0, 1, 0, 1])
        expected = hand_ibs(surv, T, E, grid, cal_T, cal_E)
        np.testing.assert_allclose(scores, [expected], rtol=RTOL, atol=ATOL)

        # pin the convention numerically: G(2) = 1 * (1 - 1/4) = 0.75 because
        # the censoring at t=2 happens after events at t=2; the event at T=2 is
        # weighted by 1/G(2), and the subject censored at T=2 contributes only
        # for grid times < 2.
        g2 = 1.0 - 1.0 / 4.0
        t = 2.0
        manual = (
            surv(t, 0) ** 2 / g2  # event at t=2
            + (1.0 - surv(t, 2)) ** 2 / g2  # subject with T=4.5 still at risk
            + (1.0 - surv(t, 3)) ** 2 / g2  # subject with T=6.0 still at risk
        ) / 4.0
        # subject 1 (censored at 2) contributes nothing at t=2
        assert abs(manual - (surv(t, 0) ** 2 / g2 + (1.0 - surv(t, 2)) ** 2 / g2 + (1.0 - surv(t, 3)) ** 2 / g2) / 4.0) < 1e-12


# ------------------------------------------------------- cross validation
class RecordingFitter(FixedCurveFitter):
    """Records the row labels seen by every fit and predict call."""

    def __init__(self):
        super().__init__(rates=np.full(64, 0.2))

    def predict_survival_function(self, X, times=None, conditional_after=None, calibration=None):
        self.predict_calls.append(list(X.index))
        return super().predict_survival_function(X, times=times, calibration=calibration)


class RecordingCalibrator(SurvivalCurveCalibrator):
    """Records the row labels seen by fit."""

    def __init__(self, log):
        super().__init__()
        self._log = log

    def fit(self, model, calibration_df, *, times):
        self._log.append(list(calibration_df.index))
        return super().fit(model, calibration_df, times=times)


class TestCalibrationCrossValidation:
    CV_GRID = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def df(self):
        cohort = make_cohort(SEED_TRAIN, 128)
        cohort.index = ["row%03d" % i for i in range(128)]
        return cohort

    @pytest.fixture
    def folds(self):
        return [
            (np.arange(0, 60), np.arange(60, 96), np.arange(96, 128)),
            (np.arange(64, 128), np.arange(0, 32), np.arange(32, 64)),
        ]

    def test_scores_are_returned_per_model_per_fold(self, df, folds):
        scores = k_fold_cross_validation(
            [CoxPHFitter(), WeibullAFTFitter()],
            df,
            "T",
            "E",
            scoring_method="integrated_brier_score",
            calibration_cv=iter(folds),
            calibrator_factory=SurvivalCurveCalibrator,
            calibration_times=self.CV_GRID,
        )
        assert len(scores) == 2
        assert all(len(model_scores) == 2 for model_scores in scores)
        assert all(score <= 0 for model_scores in scores for score in model_scores)

    def test_once_only_iterable_serves_multiple_models(self, df, folds):
        def one_pass():
            yield folds[0]
            yield folds[1]

        scores = k_fold_cross_validation(
            [CoxPHFitter(), WeibullAFTFitter()],
            df,
            "T",
            "E",
            scoring_method="integrated_brier_score",
            calibration_cv=one_pass(),
            calibrator_factory=SurvivalCurveCalibrator,
            calibration_times=self.CV_GRID,
        )
        assert all(len(model_scores) == 2 for model_scores in scores)

    def test_data_responsibilities_are_respected(self, df, folds):
        fitter = RecordingFitter()
        calibrator_fits = []
        factory_calls = []

        def factory():
            factory_calls.append(1)
            return RecordingCalibrator(calibrator_fits)

        k_fold_cross_validation(
            fitter,
            df,
            "T",
            "E",
            scoring_method="integrated_brier_score",
            calibration_cv=iter(folds),
            calibrator_factory=factory,
            calibration_times=self.CV_GRID,
        )
        labels = list(df.index)
        assert len(fitter.fit_calls) == 2
        assert len(calibrator_fits) == 2
        # a fresh calibrator per fold
        assert len(factory_calls) == 2
        for k, (train_ix, cal_ix, test_ix) in enumerate(folds):
            assert fitter.fit_calls[k] == [labels[i] for i in train_ix]
            assert calibrator_fits[k] == [labels[i] for i in cal_ix]
        # per fold the model predicts once on calibrate rows (inside
        # calibrator.fit) and once on test rows (scoring) -- never the reverse
        assert len(fitter.predict_calls) == 4
        for k, (train_ix, cal_ix, test_ix) in enumerate(folds):
            assert fitter.predict_calls[2 * k] == [labels[i] for i in cal_ix]
            assert fitter.predict_calls[2 * k + 1] == [labels[i] for i in test_ix]

    def test_new_arguments_must_come_together(self, df, folds):
        base = dict(scoring_method="integrated_brier_score")
        with pytest.raises(ValueError):
            k_fold_cross_validation(CoxPHFitter(), df, "T", "E", calibration_cv=iter(folds), **base)
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(), df, "T", "E", calibrator_factory=SurvivalCurveCalibrator, **base
            )
        with pytest.raises(ValueError):
            k_fold_cross_validation(CoxPHFitter(), df, "T", "E", calibration_times=self.CV_GRID, **base)

    def test_non_default_k_rejected_in_calibration_mode(self, df, folds):
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(),
                df,
                "T",
                "E",
                k=3,
                scoring_method="integrated_brier_score",
                calibration_cv=iter(folds),
                calibrator_factory=SurvivalCurveCalibrator,
                calibration_times=self.CV_GRID,
            )

    def test_calibration_mode_requires_integrated_brier_score(self, df, folds):
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(),
                df,
                "T",
                "E",
                scoring_method="log_likelihood",
                calibration_cv=iter(folds),
                calibrator_factory=SurvivalCurveCalibrator,
                calibration_times=self.CV_GRID,
            )

    def test_integrated_brier_score_rejected_in_legacy_mode(self, df):
        with pytest.raises(ValueError):
            k_fold_cross_validation(CoxPHFitter(), df, "T", "E", scoring_method="integrated_brier_score")

    def test_invalid_splits_rejected(self, df):
        good = dict(
            scoring_method="integrated_brier_score",
            calibrator_factory=SurvivalCurveCalibrator,
            calibration_times=self.CV_GRID,
        )
        n = len(df)
        bad_folds = [
            [(np.arange(0, 30), np.arange(30, 46), np.arange(46, 64)), ([0, 0], [1], [2])],  # duplicate in group
            [(np.arange(0, 30), np.arange(30, 46), [])],  # empty group
            [([0, 1], [1, 2], [3, 4])],  # overlap
            [([0, 1], [2, 3], [n])],  # out of bounds
            [([0, 1], [2, 3], [-1])],  # negative position
            [([0, 1], [2, 3], [4.5, 5.5])],  # non-integer
            [([0, 1], [2, 3])],  # not a triple
        ]
        for folds in bad_folds:
            with pytest.raises(ValueError):
                k_fold_cross_validation(CoxPHFitter(), df, "T", "E", calibration_cv=iter(folds), **good)

    def test_factory_must_return_fresh_unfitted_calibrators(self, df, folds):
        base = dict(scoring_method="integrated_brier_score", calibration_times=self.CV_GRID)

        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(), df, "T", "E", calibration_cv=iter(folds), calibrator_factory=lambda: object(), **base
            )

        shared = SurvivalCurveCalibrator()
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(), df, "T", "E", calibration_cv=iter(folds), calibrator_factory=lambda: shared, **base
            )

        model = CoxPHFitter().fit(df, "T", "E")
        prefitted = SurvivalCurveCalibrator().fit(model, df, times=GRID)
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(), df, "T", "E", calibration_cv=iter(folds), calibrator_factory=lambda: prefitted, **base
            )

    def test_fold_without_events_fails_with_explanation(self, df):
        no_events = df.copy()
        folds = [(np.arange(0, 30), np.arange(30, 46), np.arange(46, 64))]
        no_events.iloc[30:46, no_events.columns.get_loc("E")] = 0.0
        with pytest.raises(RuntimeError):
            k_fold_cross_validation(
                CoxPHFitter(),
                no_events,
                "T",
                "E",
                scoring_method="integrated_brier_score",
                calibration_cv=iter(folds),
                calibrator_factory=SurvivalCurveCalibrator,
                calibration_times=self.CV_GRID,
            )

    def test_fold_without_censoring_support_fails_with_explanation(self, df):
        # every calibrate subject is censored before the end of the grid
        collapsed = df.copy()
        collapsed.iloc[30:46, collapsed.columns.get_loc("E")] = 0.0
        collapsed.iloc[30:46, collapsed.columns.get_loc("T")] = 0.5
        folds = [(np.arange(0, 30), np.arange(30, 46), np.arange(46, 64))]
        with pytest.raises(RuntimeError):
            k_fold_cross_validation(
                CoxPHFitter(),
                collapsed,
                "T",
                "E",
                scoring_method="integrated_brier_score",
                calibration_cv=iter(folds),
                calibrator_factory=SurvivalCurveCalibrator,
                calibration_times=self.CV_GRID,
            )

    def test_legacy_mode_is_unchanged(self, df):
        scores = k_fold_cross_validation(CoxPHFitter(), df, "T", "E", k=4, scoring_method="concordance_index", seed=1)
        assert len(scores) == 4
        assert all(np.isfinite(s) for s in scores)

    def test_duplicate_pandas_labels_are_handled_positionally(self, df, folds):
        df = df.copy()
        df.index = ["dup"] * len(df)  # every row shares one label
        scores = k_fold_cross_validation(
            CoxPHFitter(),
            df,
            "T",
            "E",
            scoring_method="integrated_brier_score",
            calibration_cv=iter(folds),
            calibrator_factory=SurvivalCurveCalibrator,
            calibration_times=self.CV_GRID,
        )
        assert len(scores) == 2
        assert all(np.isfinite(s) for s in scores)

    def test_malformed_df_rejected(self, folds):
        good = dict(
            scoring_method="integrated_brier_score",
            calibrator_factory=SurvivalCurveCalibrator,
            calibration_times=self.CV_GRID,
        )
        df = make_cohort(SEED_TRAIN, 128)
        with pytest.raises(ValueError):
            k_fold_cross_validation(CoxPHFitter(), df.iloc[0:0], "T", "E", calibration_cv=iter(folds), **good)
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(), df.drop(columns=["T"]), "T", "E", calibration_cv=iter(folds), **good
            )
        with pytest.raises(ValueError):
            k_fold_cross_validation(
                CoxPHFitter(), df.drop(columns=["E"]), "T", "E", calibration_cv=iter(folds), **good
            )
        bad = df.copy()
        bad.loc[bad.index[0], "E"] = 3
        with pytest.raises(ValueError):
            k_fold_cross_validation(CoxPHFitter(), bad, "T", "E", calibration_cv=iter(folds), **good)
        bad = df.copy()
        bad.loc[bad.index[0], "T"] = np.nan
        with pytest.raises(ValueError):
            k_fold_cross_validation(CoxPHFitter(), bad, "T", "E", calibration_cv=iter(folds), **good)


# -------------------------------------------- independent float cross-check
class TestFloatCrossCheck:
    def test_public_entry_matches_independent_reference(self, miscalibrated):
        models, calibrators, _, test = miscalibrated
        for key in ("cph", "aft"):
            model, calibrator = models[key], calibrators[key]
            times = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
            public = model.predict_survival_function(test, times=times, calibration=calibrator).to_numpy()
            raw = model.predict_survival_function(test, times=times).to_numpy()
            # independent re-implementation of the map from the stored state
            u = np.log(np.clip(-np.log(np.clip(raw, 1e-300, 1.0)), 1e-300, None))
            knots = calibrator._knots
            slopes = np.exp(calibrator._log_slopes)
            starts = np.concatenate(([0.0], knots))
            values = np.concatenate(
                ([calibrator._intercept], calibrator._intercept + np.cumsum(slopes[:-1] * np.diff(starts)))
            )
            idx = np.searchsorted(knots, u, side="right")
            phi = values[idx] + slopes[idx] * (u - starts[idx])
            reference = np.exp(-np.exp(phi))
            np.testing.assert_allclose(public, reference, rtol=RTOL, atol=ATOL)


# -------------------------------------- heavy censoring and sparse tails
class TestAdversarialCensoring:
    def test_heavy_censoring_still_yields_valid_curves(self):
        train = make_cohort(SEED_TRAIN, 400)
        cal = make_cohort(SEED_CAL, 300, censor_rate=0.8)  # ~90% censored
        assert cal["E"].mean() < 0.25
        model = CoxPHFitter().fit(train, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
        out = model.predict_survival_function(cal, calibration=calibrator).to_numpy()
        assert np.all(out >= 0) and np.all(out <= 1)
        assert np.all(np.diff(out, axis=0) <= 1e-12)

    def test_sparse_tail_still_yields_valid_curves(self):
        train = make_cohort(SEED_TRAIN, 400)
        cal = make_cohort(SEED_CAL, 300)
        # grid reaches far beyond the bulk of events
        wide_grid = np.array([1.0, 3.0, 6.0, 12.0, 24.0])
        model = CoxPHFitter().fit(train, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=wide_grid)
        out = model.predict_survival_function(cal, calibration=calibrator).to_numpy()
        assert np.all(out >= 0) and np.all(out <= 1)
        assert np.all(np.diff(out, axis=0) <= 1e-12)

    def test_scale_miscalibration_is_corrected(self):
        # train and test differ only by a constant hazard multiplier
        train = make_cohort(SEED_TRAIN, N_TRAIN, intercept=2.4, shape=1.6)
        cal = make_cohort(SEED_CAL, N_CAL)
        test = make_cohort(SEED_TEST, N_TEST)
        model = WeibullAFTFitter().fit(train, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
        assert ipcw_ibs(model, test, GRID, calibration=calibrator) < ipcw_ibs(model, test, GRID) - 0.01

    def test_time_varying_miscalibration_is_corrected(self):
        # train and test differ in the Weibull shape (hazard ratio changes over time)
        train = make_cohort(SEED_TRAIN, N_TRAIN, intercept=1.8, shape=1.0)
        cal = make_cohort(SEED_CAL, N_CAL)
        test = make_cohort(SEED_TEST, N_TEST)
        model = WeibullAFTFitter().fit(train, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
        assert ipcw_ibs(model, test, GRID, calibration=calibrator) < ipcw_ibs(model, test, GRID) - 0.005


# ------------------------------------------------------- miscellaneous scope
class TestScope:
    def test_model_fitted_without_event_column_treats_all_as_events(self):
        df = make_cohort(SEED_TRAIN, 400)
        df["E"] = 1.0
        cal = make_cohort(SEED_CAL, 200)
        cal["E"] = 1.0
        model = WeibullAFTFitter().fit(df.drop(columns=["E"]), "T")
        calibrator = SurvivalCurveCalibrator().fit(model, cal.drop(columns=["E"]), times=GRID)
        assert calibrator.n_events_ == len(cal)
        out = model.predict_survival_function(cal.drop(columns=["E"]), calibration=calibrator)
        assert out.shape == (len(GRID), len(cal))

    def test_aft_with_ancillary_covariates(self):
        df = make_cohort(SEED_TRAIN, 400)
        cal = make_cohort(SEED_CAL, 200)
        model = WeibullAFTFitter(model_ancillary=True).fit(df, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(model, cal, times=GRID)
        out = model.predict_survival_function(cal, calibration=calibrator)
        assert out.shape == (len(GRID), len(cal))
        out_anc = model.predict_survival_function(cal, ancillary=cal, calibration=calibrator)
        np.testing.assert_allclose(out.to_numpy(), out_anc.to_numpy(), rtol=RTOL, atol=ATOL)

    def test_constructor_rejects_bad_hyperparameters(self):
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator(penalizer=-1.0)
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator(penalizer=np.inf)
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator(max_segments=0)
        with pytest.raises(ValueError):
            SurvivalCurveCalibrator(min_events_per_segment=0)
