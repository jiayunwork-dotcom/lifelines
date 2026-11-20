# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines.utils import CensoringType
from lifelines.fitters import RegressionFitter
from lifelines import CRCSplineFitter


def _validate_calibration_times(times):
    """
    Validate a calibration time grid: at least three strictly increasing,
    finite, positive time points. Returns a float ndarray copy.
    """
    try:
        arr = np.asarray(list(times), dtype=float)
    except (TypeError, ValueError):
        raise ValueError("times must be an iterable of at least three strictly increasing finite positive numbers")
    if arr.ndim != 1 or arr.size < 3:
        raise ValueError("times must contain at least three time points")
    if not np.all(np.isfinite(arr)):
        raise ValueError("times must all be finite")
    if np.any(arr <= 0):
        raise ValueError("times must all be strictly positive")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("times must be strictly increasing")
    return arr


class SurvivalCurveCalibrator:
    r"""
    Learn a reusable recalibration of the *entire* survival curve produced by a
    fitted lifelines regression model, using an independent, right-censored
    follow-up cohort.

    This is **not** a collection of per-time-point binary calibrators. A single
    monotone map is learned jointly from all follow-up times and is then applied
    to any query time inside the calibration domain, so the calibrated output is
    again a proper survival curve.

    Statistical model
    -----------------
    Write the fitted model's predicted cumulative hazard for a subject as
    ``h(t) = -log S_model(t | x)`` and let ``u = log h(t)`` (this is the
    complementary log-log scale of the predicted event probability, the same
    scale used by :func:`survival_probability_calibration`). The calibrator
    learns a continuous, piecewise-linear, strictly increasing map ``phi`` and
    defines the calibrated curve as

    .. math:: S_{cal}(t \mid x) = \exp\!\left(-\exp\!\left(\phi(\log h(t \mid x))\right)\right)

    Because ``phi`` is strictly increasing and shared by all subjects and all
    times:

    * ``S_cal`` lies in (0, 1] and is non-increasing in ``t`` over the whole
      calibration domain;
    * at any fixed ``t``, subjects ordered by model risk keep their order
      (calibration never reverses two subjects at the same time), and identical
      raw curves receive identical calibrated curves;
    * the map is defined at any query time, not only at the grid points.

    Estimation under censoring
    --------------------------
    ``phi`` is estimated by maximizing the likelihood of the calibration cohort
    under the calibrated curves. Censoring is **never** recoded as "no event":

    * an event observed in grid interval ``(t_{j-1}, t_j]`` contributes the
      calibrated interval mass ``S_cal(t_{j-1}) - S_cal(t_j)`` (a grouped-time /
      interval-censored contribution, which also accommodates the discrete
      Breslow baseline of a Cox model);
    * a subject censored at ``u`` contributes ``S_cal(u)`` evaluated at the
      exact censoring time;
    * any follow-up beyond the last grid point ``t_K`` is administratively
      censored at ``t_K`` (contributes ``S_cal(t_K)``): no curve shape is
      claimed outside the calibration domain.

    Complexity control
    ------------------
    The number of linear segments is
    ``min(max_segments, len(times) - 1, n_events // min_events_per_segment)``
    with knots at quantiles of the pooled log-cumulative-hazards on the grid,
    and a ridge penalty ``penalizer * (intercept^2 + sum(log-slope^2))`` shrinks
    ``phi`` towards the identity map (i.e. towards "the model is already
    calibrated"). The trade-off: more segments can represent time-varying
    miscalibration, but each segment is identified only by the events whose
    interval falls in its hazard range. With heavy censoring or few events the
    likelihood is weak and the penalty dominates, pulling the map back to the
    identity instead of overfitting noise; with rich event information the
    likelihood dominates and genuine miscalibration is learned. A grid that is
    too fine leaves intervals without events (locally flat likelihood, so the
    penalty wins locally); a grid that is too coarse cannot resolve where in
    time the miscalibration changes.

    Assumptions and scope
    ---------------------
    * Single-event, right-censored data with follow-up starting at 0.
    * Numeric covariates only. Models fitted with weights, entry times
      (left truncation) or strata are rejected, as are competing risks and
      conditional-survival (``conditional_after``) predictions.
    * Supports :class:`~lifelines.CoxPHFitter` (including its default Breslow
      baseline) and the AFT fitters such as :class:`~lifelines.WeibullAFTFitter`.
    * The calibration cohort must be **independent of the model's training
      data**. This class never refits the base model; it is the caller's
      responsibility to supply a genuinely separate cohort. No attempt is made
      to detect overlap from data values.
    * Predictions are only a function of the learned map and the subject's raw
      model curve; outcomes of the subjects being predicted are never read.

    Time domain and extrapolation
    -----------------------------
    Calibrated predictions are defined only for query times inside
    ``[times[0], times[-1]]``. Queries outside that interval, or non-finite or
    non-positive times, are rejected with ``ValueError`` rather than
    extrapolated.

    Limitations
    -----------
    * Calibration is learned from a finite, censored sample: there is **no
      guarantee** that calibrated probabilities are closer to the true
      distribution on arbitrary data-generating processes. If the calibration
      cohort is not representative of the deployment population, recalibration
      can make probabilities worse.
    * The map is monotone on the log-cumulative-hazard scale; miscalibration
      that is not representable by such a map (e.g. crossing survival curves
      that the base model orders incorrectly) cannot be repaired.
    * The object is bound to the exact fitted model instance. Passing it to
      another model, or refitting the model after calibration, raises
      ``ValueError``.

    Parameters
    ----------
    penalizer: float, optional (default=1.0)
        Ridge penalty shrinking the learned map towards the identity
        (no recalibration). Larger values are more conservative.
    max_segments: int, optional (default=4)
        Maximum number of linear segments of the map on the
        log-cumulative-hazard scale.
    min_events_per_segment: int, optional (default=15)
        Minimum number of observed events required to support each segment;
        fewer events automatically reduce the number of segments.

    Examples
    --------
    .. code:: python

        from lifelines import CoxPHFitter, SurvivalCurveCalibrator

        cph = CoxPHFitter().fit(train_df, "T", "E")
        calibrator = SurvivalCurveCalibrator().fit(cph, calibration_df, times=[5, 10, 15, 20, 25])
        calibrated_curves = cph.predict_survival_function(new_df, calibration=calibrator)

    """

    def __init__(self, penalizer: float = 1.0, max_segments: int = 4, min_events_per_segment: int = 15):
        if not np.isscalar(penalizer) or not np.isfinite(penalizer) or penalizer < 0:
            raise ValueError("penalizer must be a finite, non-negative number")
        if not isinstance(max_segments, (int, np.integer)) or isinstance(max_segments, bool) or max_segments < 1:
            raise ValueError("max_segments must be a positive integer")
        if (
            not isinstance(min_events_per_segment, (int, np.integer))
            or isinstance(min_events_per_segment, bool)
            or min_events_per_segment < 1
        ):
            raise ValueError("min_events_per_segment must be a positive integer")
        self.penalizer = float(penalizer)
        self.max_segments = int(max_segments)
        self.min_events_per_segment = int(min_events_per_segment)
        self._fitted = False

    # ------------------------------------------------------------------ fit

    def fit(self, model, calibration_df: pd.DataFrame, *, times) -> "SurvivalCurveCalibrator":
        """
        Learn the recalibration map from an independent calibration cohort.

        Parameters
        ----------
        model:
            an already fitted :class:`~lifelines.CoxPHFitter` or lifelines AFT
            fitter (e.g. :class:`~lifelines.WeibullAFTFitter`). The model is
            inspected but never modified or refitted.
        calibration_df: DataFrame
            follow-up data of a cohort independent of the model's training
            data. Must contain the model's ``duration_col`` and (if the model
            was fitted with one) ``event_col``; the event column must be coded
            0/1. If the model was fitted without an event column, every
            subject is treated as having an observed event.
        times:
            at least three strictly increasing, finite, positive time points.
            They define the calibration domain and the grouped-time likelihood
            intervals, and become the default prediction grid.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            for unsupported models, empty or malformed calibration data,
            non-finite values, non-0/1 events, or invalid ``times``.
        RuntimeError
            if the calibration likelihood cannot be estimated (e.g. no
            observed events, or numerical failure of the optimizer).
        """
        from lifelines.fitters import ParametricRegressionFitter
        from lifelines.fitters.coxph_fitter import CoxPHFitter

        # ---- model checks -------------------------------------------------
        if not isinstance(model, (CoxPHFitter, ParametricRegressionFitter)):
            raise ValueError(
                "SurvivalCurveCalibrator supports CoxPHFitter and lifelines AFT fitters "
                "(e.g. WeibullAFTFitter); got %s" % type(model).__name__
            )
        if getattr(model, "params_", None) is None:
            raise ValueError("model is not fitted; fit the model before calibrating it")
        if not CensoringType.is_right_censoring(model):
            raise ValueError("only right-censored, single-event models are supported")
        if getattr(model, "weights_col", None) is not None:
            raise ValueError("models fitted with weights are not supported")
        if getattr(model, "entry_col", None) is not None:
            raise ValueError("models fitted with entry times (left truncation) are not supported")
        strata = getattr(model, "strata", None)
        if strata is not None and len(strata) > 0:
            raise ValueError("stratified models are not supported")

        # ---- times --------------------------------------------------------
        grid = _validate_calibration_times(times)
        t_max = grid[-1]

        # ---- calibration data ---------------------------------------------
        if not isinstance(calibration_df, pd.DataFrame):
            raise ValueError("calibration_df must be a pandas DataFrame")
        if calibration_df.shape[0] == 0:
            raise ValueError("calibration_df is empty")

        duration_col = model.duration_col
        event_col = model.event_col  # may be None -> all events observed

        if duration_col not in calibration_df.columns:
            raise ValueError("duration column '%s' is not present in calibration_df" % duration_col)
        if event_col is not None and event_col not in calibration_df.columns:
            raise ValueError("event column '%s' is not present in calibration_df" % event_col)

        try:
            durations = calibration_df[duration_col].to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise ValueError("durations must be numeric")
        if not np.all(np.isfinite(durations)):
            raise ValueError("durations must all be finite")
        if np.any(durations < 0):
            raise ValueError("durations must be non-negative")

        if event_col is None:
            events = np.ones(calibration_df.shape[0], dtype=float)
        else:
            try:
                events = calibration_df[event_col].to_numpy(dtype=float)
            except (TypeError, ValueError):
                raise ValueError("event column must be numeric 0/1")
            if not np.all(np.isfinite(events)) or not np.all((events == 0) | (events == 1)):
                raise ValueError("event column must be coded 0/1")

        covariate_cols = [c for c in calibration_df.columns if c not in (duration_col, event_col)]
        for c in covariate_cols:
            if not pd.api.types.is_numeric_dtype(calibration_df[c]):
                raise ValueError("covariate column '%s' is not numeric; only numeric covariates are supported" % c)
            if not np.all(np.isfinite(calibration_df[c].to_numpy(dtype=float))):
                raise ValueError("covariate column '%s' contains non-finite values" % c)

        n_events = int(events.sum())
        if n_events == 0:
            raise RuntimeError(
                "cannot estimate a calibration map: the calibration cohort has no observed events"
            )

        # ---- raw model curves on the grid and at censoring times ----------
        censored_in_domain = (events == 0) & (durations > 0) & (durations <= t_max)
        eval_times = np.union1d(grid, durations[censored_in_domain])
        raw = model.predict_survival_function(calibration_df, times=eval_times)
        raw_values = np.asarray(raw, dtype=float)  # rows: eval_times, cols: subjects
        if raw_values.shape != (eval_times.size, calibration_df.shape[0]):
            raise RuntimeError(
                "unexpected shape %s of model predictions for %d subjects"
                % (raw_values.shape, calibration_df.shape[0])
            )
        u_all = self._survival_to_u(raw_values)  # log cumulative hazard scale
        u_grid = u_all[np.searchsorted(eval_times, grid), :]
        if censored_in_domain.any():
            u_cens = u_all[np.searchsorted(eval_times, durations[censored_in_domain]), np.where(censored_in_domain)[0]]
        else:
            u_cens = np.empty(0)

        # ---- complexity control: number of segments and knots -------------
        n_segments = int(
            max(1, min(self.max_segments, grid.size - 1, n_events // self.min_events_per_segment))
        )
        knots = np.empty(0)
        if n_segments > 1:
            pooled = u_grid.ravel()
            # ignore the artificial floor of clipped (essentially zero) hazards
            # when placing knots, so knots land where the likelihood has support
            finite_pooled = pooled[pooled > -600.0]
            if finite_pooled.size == 0:
                finite_pooled = pooled
            quantiles = np.quantile(finite_pooled, np.linspace(0.0, 1.0, n_segments + 1)[1:-1])
            knots = np.unique(quantiles)
            n_segments = knots.size + 1

        # ---- grouped-time negative log-likelihood -------------------------
        # interval index k in 0..K for events: (t_{k-1}, t_k] with t_{-1} := 0,
        # k == K means "beyond the grid" (administratively censored at t_K).
        event_interval = np.searchsorted(grid, durations, side="left")
        K = grid.size

        def neg_log_likelihood(params):
            intercept = params[0]
            slopes = np.exp(params[1:])
            s_grid = self._map_u_to_survival(u_grid, knots, intercept, slopes)  # (K, n)
            log_s_grid = -np.exp(self._phi(u_grid, knots, intercept, slopes))  # = log S_cal
            nll = 0.0
            # events inside the grid: interval mass S(t_{k-1}) - S(t_k)
            inside = (events == 1) & (event_interval < K)
            if inside.any():
                k = event_interval[inside]
                rows = np.where(inside)[0]
                s_hi = s_grid[k, rows]
                s_lo = np.where(k == 0, 1.0, s_grid[np.maximum(k - 1, 0), rows])
                mass = np.clip(s_lo - s_hi, 1e-12, 1.0)
                nll -= np.log(mass).sum()
            # events beyond the grid: survived the domain
            beyond = (events == 1) & (event_interval >= K)
            if beyond.any():
                nll -= log_s_grid[K - 1, np.where(beyond)[0]].sum()
            # censored beyond the grid: survived the domain
            cens_beyond = (events == 0) & (durations > t_max)
            if cens_beyond.any():
                nll -= log_s_grid[K - 1, np.where(cens_beyond)[0]].sum()
            # censored inside the domain: exact S_cal(u)
            if censored_in_domain.any():
                log_s_cens = -np.exp(self._phi(u_cens, knots, intercept, slopes))
                nll -= log_s_cens.sum()
            # ridge penalty towards the identity map
            nll += self.penalizer * (intercept**2 + np.sum(params[1:] ** 2))
            return nll

        from scipy.optimize import minimize

        x0 = np.zeros(n_segments + 1)
        try:
            result = minimize(neg_log_likelihood, x0, method="BFGS", options={"maxiter": 500})
            if not (result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun)):
                result = minimize(
                    neg_log_likelihood, x0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8}
                )
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("numerical estimation of the calibration map failed: %s" % exc)
        if not (np.all(np.isfinite(result.x)) and np.isfinite(result.fun)):
            raise RuntimeError(
                "numerical estimation of the calibration map failed: optimizer returned a non-finite solution"
            )

        # ---- commit learned state (only on success) -----------------------
        self._knots = knots
        self._intercept = float(result.x[0])
        self._log_slopes = np.asarray(result.x[1:], dtype=float)
        self.times_ = grid.copy()
        self.duration_col_ = duration_col
        self.event_col_ = event_col
        self.n_events_ = n_events
        self.n_subjects_ = calibration_df.shape[0]
        # CoxPHFitter is a delegating facade around its fitted internal model
        # (predictions are served by the internal object), so bind to the
        # object that actually serves predict_survival_function calls.
        self._model = model._model if isinstance(model, CoxPHFitter) else model
        self._model_params_ref = self._model.params_  # pins the id() used in the fingerprint
        self._model_fingerprint = self._fingerprint(self._model)
        self._fitted = True
        return self

    # ------------------------------------------------- internal mathematics

    @staticmethod
    def _survival_to_u(s):
        s = np.clip(np.asarray(s, dtype=float), 1e-300, 1.0)
        h = -np.log(s)
        return np.log(np.clip(h, 1e-300, None))

    @staticmethod
    def _phi(u, knots, intercept, slopes):
        """Continuous piecewise-linear strictly increasing map, phi(0) = intercept."""
        u = np.asarray(u, dtype=float)
        if knots.size == 0:
            return intercept + slopes[0] * u
        idx = np.searchsorted(knots, u, side="right")
        # values of phi at the segment start points (anchor 0, then knots)
        starts = np.concatenate(([0.0], knots))
        values = np.concatenate(([intercept], intercept + np.cumsum(slopes[:-1] * np.diff(starts))))
        return values[idx] + slopes[idx] * (u - starts[idx])

    @classmethod
    def _map_u_to_survival(cls, u, knots, intercept, slopes):
        return np.exp(-np.exp(cls._phi(u, knots, intercept, slopes)))

    def _transform_survival(self, s):
        """Apply the learned map to raw survival probabilities (elementwise)."""
        u = self._survival_to_u(s)
        return self._map_u_to_survival(u, self._knots, self._intercept, np.exp(self._log_slopes))

    @staticmethod
    def _fingerprint(model):
        params = np.asarray(model.params_, dtype=float)
        # id(params_) changes on every refit because fitters reassign params_;
        # the calibrator keeps a reference to the fitted-time params_ object so
        # the id cannot be recycled by the garbage collector.
        return (model._time_fit_was_called, id(model.params_), params.shape, params.tobytes())

    def _check_bound_model(self, model):
        if not self._fitted:
            raise ValueError("this SurvivalCurveCalibrator is not fitted yet")
        if model is not self._model:
            raise ValueError(
                "this calibrator is bound to the model instance it was fitted with; "
                "it cannot be applied to a different model"
            )
        if self._model_fingerprint != self._fingerprint(model):
            raise ValueError(
                "the bound model was refitted after this calibrator was fitted; "
                "fit a new SurvivalCurveCalibrator instead of reusing the stale one"
            )

    # ------------------------------------------------------------- predict

    def _predict_calibrated(self, model, df, *, times, conditional_after, ancillary=None):
        if conditional_after is not None:
            raise ValueError("conditional_after is not supported together with calibration")
        self._check_bound_model(model)

        if times is None:
            requested = self.times_
        else:
            try:
                requested = np.atleast_1d(np.asarray(times, dtype=float))
            except (TypeError, ValueError):
                raise ValueError("times must be an iterable of finite times")
            if requested.ndim != 1 or requested.size == 0:
                raise ValueError("times must be a non-empty one-dimensional collection")
            if not np.all(np.isfinite(requested)):
                raise ValueError("times must all be finite")
            if np.any(requested < self.times_[0]) or np.any(requested > self.times_[-1]):
                raise ValueError(
                    "calibrated predictions are only defined on the calibration domain "
                    "[%g, %g]; got a query outside it" % (self.times_[0], self.times_[-1])
                )

        # evaluate the raw model on a sorted, de-duplicated grid, then restore
        # the requested order (queries may be shuffled and/or repeated).
        unique_times, inverse = np.unique(requested, return_inverse=True)
        if ancillary is None:
            raw = model.predict_survival_function(df, times=unique_times)
        else:
            raw = model.predict_survival_function(df, times=unique_times, ancillary=ancillary)
        calibrated = self._transform_survival(np.asarray(raw, dtype=float))
        calibrated = calibrated[inverse]
        return pd.DataFrame(calibrated, index=requested, columns=raw.columns)


def survival_probability_calibration(
    model: RegressionFitter, df: pd.DataFrame, t0: float, ax=None, calibration: SurvivalCurveCalibrator = None
):
    r"""
    Smoothed calibration curves for time-to-event models. This is analogous to
    calibration curves for classification models, extended to handle survival probabilities
    and censoring. Produces a matplotlib figure and some metrics.

    We want to calibrate our model's prediction of :math:`P(T < \text{t0})` against the observed frequencies.

    Parameters
    -------------

    model:
        a fitted lifelines regression model to be evaluated
    df: DataFrame
        a DataFrame - if equal to the training data, then this is an in-sample calibration. Could also be an out-of-sample
        dataset.
    t0: float
        the time to evaluate the probability of event occurring prior at.
    calibration: SurvivalCurveCalibrator, optional
        a fitted :class:`SurvivalCurveCalibrator` bound to ``model``. When
        provided, the *already recalibrated* probabilities
        (``model.predict_survival_function(df, times=[t0], calibration=calibration)``)
        are evaluated instead of the raw ones, so the plot and the ICI/E50
        describe the calibrated model. The calibrator is only read from: it is
        never refitted or modified, and the smoothing spline fitted here for
        diagnostics is not fed back into the calibrator or the model.

    Returns
    ----------
    ax:
        mpl axes
    ICI:
        mean absolute difference between predicted and observed
    E50:
        median absolute difference between predicted and observed

    https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8570

    """

    def ccl(p):
        return np.log(-np.log(1 - p))

    if ax is None:
        ax = plt.gca()

    T = model.duration_col
    E = model.event_col

    if calibration is None:
        predictions_at_t0 = np.clip(1 - model.predict_survival_function(df, times=[t0]).T.squeeze(), 1e-10, 1 - 1e-10)
    else:
        predictions_at_t0 = np.clip(
            1 - model.predict_survival_function(df, times=[t0], calibration=calibration).T.squeeze(), 1e-10, 1 - 1e-10
        )

    # create new dataset with the predictions
    prediction_df = pd.DataFrame({"ccl_at_%d" % t0: ccl(predictions_at_t0), T: df[T], E: df[E]})

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    n_knots = 3
    regressors = {"beta_": ["ccl_at_%d" % t0], "gamma0_": "1", "gamma1_": "1", "gamma2_": "1"}

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if CensoringType.is_right_censoring(model):
            crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_left_censoring(model):
            crc.fit_left_censoring(prediction_df, T, E, regressors=regressors)
        elif CensoringType.is_interval_censoring(model):
            crc.fit_interval_censoring(prediction_df, T, E, regressors=regressors)

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(np.clip(predictions_at_t0.min() - 0.01, 0, 1), np.clip(predictions_at_t0.max() + 0.01, 0, 1), 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}), times=[t0]).T.squeeze()

    # plot our results
    ax.set_title("Smoothed calibration curve of \npredicted vs observed probabilities of t ≤ %d mortality" % t0)

    color = "tab:red"
    ax.plot(x, y, label="smoothed calibration curve", color=color)
    ax.set_xlabel("Predicted probability of \nt ≤ %d mortality" % t0)
    ax.set_ylabel("Observed probability of \nt ≤ %d mortality" % t0, color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # plot x=y line
    ax.plot(x, x, c="k", ls="--")
    ax.legend()

    # plot histogram of our original predictions
    color = "tab:blue"
    twin_ax = ax.twinx()
    twin_ax.set_ylabel("Count of \npredicted probabilities", color=color)  # we already handled the x-label with ax1
    twin_ax.tick_params(axis="y", labelcolor=color)
    twin_ax.hist(predictions_at_t0, alpha=0.3, bins="sqrt", color=color)

    plt.tight_layout()

    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze() - predictions_at_t0).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    print("ICI = ", ICI)
    print("E50 = ", E50)

    return ax, ICI, E50
