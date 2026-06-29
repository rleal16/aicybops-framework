from enum import Enum, auto
import logging
import numpy as np
from typing import Optional

class Status(Enum):
    """
    Detection result
    """

    NORMAL = auto()
    ABNORMAL = auto()
    ALARM = auto()


class EVTThreshold:
    """
    Implements dynamic thresholding based on Extreme Value Theory using the POT method.
    """

    def __init__(
            self, 
            risk_level: float = 1e-4,
            max_candidates: int = 10,
            tail: str = "upper", # "upper" or "lower"
            logger: Optional[logging.Logger] = None,
            optimizer_options: Optional[dict] = None,
    ):
        self.risk_level = risk_level
        self.max_candidates = max_candidates
        self.tail = tail
        self._init_threshold = None
        self._peaks = np.array([])
        self._gamma = None
        self._sigma = None
        self.logger = logger or logging.getLogger(__name__)
        self._current_threshold = None  # Store the current threshold as state
        self.optimizer_options = optimizer_options or {}
        self.logger.info(f"Initializing EVTThreshold with risk_level={risk_level}, max_candidates={max_candidates}, tail={tail}")
    
    def get_risk_level(self):
        return self.risk_level

    def _grimshaw(self, peaks: np.ndarray, epsilon: float = 1e-8) -> tuple:
        """
        Compute the GPD parameters estimation with the Grimshaw's trick (from reference SPOT).
        Returns: gamma, sigma
        """
        def _u(s):
            return 1 + np.log(s).mean()
        def _v(s):
            return np.mean(1 / s)
        def _w(t):
            s = 1 + t * peaks
            us = _u(s)
            vs = _v(s)
            return us * vs - 1
        def _jac_w(t):
            s = 1 + t * peaks
            us = _u(s)
            vs = _v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s**2))
            return us * jac_vs + vs * jac_us
        y_min = peaks.min()
        y_max = peaks.max()
        y_mean = peaks.mean()
        a = -1 / y_max
        if abs(a) < 3 * epsilon:
            epsilon = abs(a) / self.max_candidates
        a = a + epsilon
        # We look for possible roots
        def _roots_finder(fun, jac, bounds, npoints, method):
            if method == "regular":
                step = (bounds[1] - bounds[0]) / (npoints + 1)
                initial_guess = np.arange(bounds[0] + step, bounds[1], step)
            elif method == "random":
                initial_guess = np.random.uniform(bounds[0], bounds[1], npoints)
            def _object(variable):
                value = np.array([fun(item) for item in variable])
                gradient = np.array([jac(item) for item in variable])
                return (value**2).sum(), 2 * value * gradient
            from scipy.optimize import minimize
            opt = minimize(
                _object,
                initial_guess,
                method="L-BFGS-B",
                jac=True,
                bounds=[bounds] * len(initial_guess),
                options=self.optimizer_options,
            )
            X = opt.x
            np.round(X, decimals=5)
            return np.unique(X)
        left_zeros = _roots_finder(
            _w,
            _jac_w,
            (a + epsilon, -epsilon),
            self.max_candidates,
            "regular",
        )
        if y_mean > y_min > 0 and not np.isclose(y_mean, y_min):
            b = 2 * (y_mean - y_min) / (y_mean * y_min)
            c = 2 * (y_mean - y_min) / (y_min**2)
            right_zeros = _roots_finder(
                _w,
                _jac_w,
                (b, c),
                self.max_candidates,
                "regular",
            )
            zeros = np.concatenate((left_zeros, right_zeros))
        else:
            zeros = left_zeros
        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = y_mean
        ll_best = self._log_likelihood(peaks, gamma_best, sigma_best)
        for z in zeros:
            gamma = _u(1 + z * peaks) - 1
            sigma = gamma / z
            ll = self._log_likelihood(peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
        return gamma_best, sigma_best

    @staticmethod
    def _log_likelihood(Y: np.ndarray, gamma: float, sigma: float) -> float:
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * np.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            # Corrected: log-likelihood for Exponential (GPD with gamma=0)
            if sigma <= 0:
                return -np.inf
            L = -n * np.log(sigma) - (Y / sigma).sum()
        return L
    
    def fit_baseline(self, data: np.ndarray, init_quantile: float = 0.95):
        """
        Fit the initial threshold and GPD to the baseline data.
        Uses a high quantile (95%) for initial threshold as per paper.
        """
        n = len(data)
        sorted_data = sorted(data)
        idx = int(init_quantile * n)
        
        self.logger.debug("EVTThreshold.fit_baseline:")
        self.logger.debug(f"  - Data points: {n}")
        self.logger.debug(f"  - Init quantile: {init_quantile}")
        self.logger.debug(f"  - Data range: [{sorted_data[0]:.4f}, {sorted_data[-1]:.4f}]")
        
        if self.tail == "upper":
            self._init_threshold = sorted_data[idx]
            self._peaks = data[data > self._init_threshold] - self._init_threshold
        else:
            self._init_threshold = sorted_data[int((1 - init_quantile) * n)]
            self._peaks = self._init_threshold - data[data < self._init_threshold]
        
        self.logger.debug(f"  - Initial threshold: {self._init_threshold:.4f}")
        self.logger.debug(f"  - Number of peaks: {len(self._peaks)}")
        if len(self._peaks) > 0:
            self.logger.debug(f"  - Peak range: [{self._peaks.min():.4f}, {self._peaks.max():.4f}]")
        
        if len(self._peaks) == 0:
            self._gamma, self._sigma = None, None
            self.logger.warning("  - WARNING: No peaks for GPD fit; using percentile threshold only.")
            self._current_threshold = self._init_threshold
        else:
            self._gamma, self._sigma = self._grimshaw(self._peaks)
            # Clamp gamma to avoid unstable GPD fits.
            _GAMMA_MAX = 0.5
            if self._gamma is not None and abs(self._gamma) > _GAMMA_MAX:
                self.logger.info(
                    "  - Clamping gamma from %.4f to [%.1f, %.1f]",
                    self._gamma, -_GAMMA_MAX, _GAMMA_MAX,
                )
                self._gamma = max(-_GAMMA_MAX, min(_GAMMA_MAX, self._gamma))
            self.logger.debug(f"  - GPD parameters - gamma: {self._gamma:.4f}, sigma: {self._sigma:.4f}")
            self._current_threshold = self._compute_quantile(num_total=n)
            # Cap threshold relative to init_threshold.
            self._baseline_max = float(data.max())
            _threshold_cap = self._init_threshold * 1.5
            if self._current_threshold > _threshold_cap:
                self.logger.info(
                    "  - Capping threshold from %.4f to %.4f (1.5x init_threshold)",
                    self._current_threshold, _threshold_cap,
                )
                self._current_threshold = _threshold_cap
            self.logger.debug(f"  - Initial dynamic threshold: {self._current_threshold:.4f}")

    def _compute_quantile(self, num_total: int):
        """
        Compute the dynamic threshold using the formula:
        Th ≃ th + β̂/γ̂((qn/Nt)^(-γ̂) - 1)
        where:
        - th: initial threshold
        - β̂, γ̂: GPD parameters
        - q: risk level
        - n: total observations
        - Nt: number of peaks
        """
        if self._gamma is None or self._sigma is None or len(self._peaks) == 0:
            self.logger.debug(f"_compute_quantile: No GPD parameters, returning init_threshold: {self._init_threshold:.4f}")
            return self._init_threshold
            
        n = num_total
        q = self.risk_level
        N_t = len(self._peaks)
        r = n * q / N_t
        
        self.logger.debug(f"_compute_quantile: n={n}, q={q}, N_t={N_t}, r={r:.6f}")
        
        if abs(self._gamma) > 1e-8:
            offset = (self._sigma / self._gamma) * ((r ** (-self._gamma)) - 1)
            threshold = self._init_threshold + offset if self.tail == "upper" else self._init_threshold - offset
            self.logger.debug(f"GPD case: offset={offset:.4f}, threshold={threshold:.4f}")
        else:
            offset = self._sigma * np.log(r)
            threshold = self._init_threshold - offset if self.tail == "upper" else self._init_threshold + offset
            self.logger.debug(f"Exponential case: offset={offset:.4f}, threshold={threshold:.4f}")
        
        return threshold
    
    def _refit_and_update_threshold(self, num_total: int):
        """Refit GPD on peaks, clamp gamma, compute new threshold with cap."""
        self._gamma, self._sigma = self._grimshaw(self._peaks)
        _GAMMA_MAX = 0.5
        if self._gamma is not None and abs(self._gamma) > _GAMMA_MAX:
            self._gamma = max(-_GAMMA_MAX, min(_GAMMA_MAX, self._gamma))
        self._current_threshold = self._compute_quantile(num_total)
        _cap = self._init_threshold * 1.5
        if self._current_threshold > _cap:
            self._current_threshold = _cap

    def process(self, score: float, num_total: int, update_with_alarm: bool = False) -> Status:
        """
        Process a new score in the stream and return its status.
        Only updates peaks/GPD for ABNORMAL (unless update_with_alarm=True for ALARM).
        """
        self.logger.debug(f"Processing score: {score:.4f}, current_threshold: {self._current_threshold:.4f}")

        status = None
        if self.tail == "upper":
            if score <= self._init_threshold:
                status = Status.NORMAL
                self.logger.debug(f"  -> Status: NORMAL (score <= init_threshold: {self._init_threshold:.4f})")
            elif score > self._current_threshold:
                status = Status.ALARM
                self.logger.debug(f"  -> Status: ALARM (score > current_threshold: {self._current_threshold:.4f})")
                if update_with_alarm:
                    self._peaks = np.append(self._peaks, score - self._init_threshold)
                    self._refit_and_update_threshold(num_total)
            else:
                status = Status.ABNORMAL
                self.logger.debug(f"  -> Status: ABNORMAL (init_threshold < score <= current_threshold)")
                self._peaks = np.append(self._peaks, score - self._init_threshold)
                self._refit_and_update_threshold(num_total)
        else:
            if score >= self._init_threshold:
                status = Status.NORMAL
            elif score < self._current_threshold:
                status = Status.ALARM
                if update_with_alarm:
                    self._peaks = np.append(self._peaks, self._init_threshold - score)
                    self._refit_and_update_threshold(num_total)
            else:
                status = Status.ABNORMAL
                self._peaks = np.append(self._peaks, self._init_threshold - score)
                self._refit_and_update_threshold(num_total)
        
        return status

    def get_current_threshold(self, num_total: int) -> float:
        return self._current_threshold

class AnomalyDetector:
    """
    Base class for anomaly detectors (SPOT, DriftSPOT).
    """
    
    def __init__(self, risk_level):
        self._baseline_fitted = False
        self._risk_level = risk_level
    
    def get_baseline_fitted(self) -> bool:
        """Returns True if the baseline has been fitted."""
        return self._baseline_fitted
    
    def get_risk_level(self) -> float:
        return self._risk_level
    
    def get_depth(self) -> Optional[int]:
        """Returns the depth parameter (for dSPOT) or None if not applicable."""
        return getattr(self, 'depth', None)
        

class SPOT(AnomalyDetector):
    """
    Streaming Peaks-Over-Threshold (SPOT) anomaly detector.
    Uses a single EVTThreshold for upper- or lower-tail detection.
    Equivalent to SPOT in ads-evt-main, but with improved state management.
    """

    def __init__(self, risk_level=1e-4, max_candidates=10, tail="upper"):
        super().__init__(risk_level=risk_level)
        self.evt = EVTThreshold(risk_level=risk_level, max_candidates=max_candidates, tail=tail)
        self.stream_scores = None

    def fit_baseline(self, baseline_scores: np.ndarray, stream_scores: np.ndarray, init_quantile: float = 0.98):
        """
        Fit the baseline (initialization) data and set up the stream.
        Only baseline_scores are used for EVTThreshold calibration.
        """
        self.baseline_scores = baseline_scores
        self.stream_scores = stream_scores
        self.evt.fit_baseline(baseline_scores, init_quantile)
        self._baseline_fitted = True

    def run_stream(self, update_with_alarm: bool = False, stream_data: np.ndarray = None):
        """
        Process the stream data, returning alarms and thresholds.
        Delegates all state management to EVTThreshold.
        """
        assert self._baseline_fitted, "Call fit_baseline() first."
        # Default to the stream provided at fit time if none is passed
        if stream_data is None:
            stream_data = self.stream_scores
        if stream_data is None:
            raise ValueError("No stream data provided to run_stream and no stored stream from fit_baseline().")
        alarms = []
        thresholds = []
        num_total = 0
        for i, score in enumerate(stream_data):
            status = self.evt.process(score, num_total + len(self.baseline_scores), update_with_alarm=update_with_alarm)
            if status == Status.ALARM:
                alarms.append(i)
            else:
                num_total += 1
            thresholds.append(self.evt.get_current_threshold(num_total + len(self.baseline_scores)))
        return {"alarms": alarms, "thresholds": thresholds}

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average for detrending.
    Equivalent to moving_average in ads-evt-main.
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')

class DriftSPOT(AnomalyDetector):
    """
    Drift SPOT (dSPOT): applies moving average detrending before SPOT.
    This is a thin wrapper around SPOT, only changing the data preprocessing.
    Equivalent to dSPOT in ads-evt-main, but with improved state management.
    """

    def __init__(self, risk_level=1e-4, max_candidates=10, depth=10):
        super().__init__(risk_level=risk_level)
        self.depth = depth
        self.spot = SPOT(risk_level=risk_level, max_candidates=max_candidates, tail="upper")
        self.raw_stream_scores = None
        self.detrended_baseline = None
        self.window = []  # Store the moving window
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing DriftSPOT with risk_level={risk_level}, depth={depth}")

    def fit_baseline(self, baseline_scores: np.ndarray, stream_scores: np.ndarray, init_quantile: float = 0.95):
        """
        Detrend baseline, then fit SPOT. Store raw stream scores for later detrending.
        The detrended baseline is baseline_scores[self.depth:] - moving_average(baseline_scores, self.depth)[:-1].
        Only the detrended baseline is used for EVTThreshold calibration.
        """
        self.logger.debug("DriftSPOT.fit_baseline:")
        self.logger.debug(f"  - Baseline scores: {len(baseline_scores)} points")
        self.logger.debug(f"  - Stream scores: {len(stream_scores)} points")
        self.logger.debug(f"  - Depth: {self.depth}")
        
        self.baseline_scores = np.asarray(baseline_scores)
        n = len(self.baseline_scores)
        if n <= self.depth:
            self.detrended_baseline = self.baseline_scores.copy()
            self.window = list(self.baseline_scores) if n else []
        else:
            ma = moving_average(self.baseline_scores, self.depth)[:-1]
            tail = self.baseline_scores[self.depth:]
            if len(tail) != len(ma):
                self.detrended_baseline = self.baseline_scores.copy()
                self.window = list(self.baseline_scores[-self.depth:])
            else:
                self.detrended_baseline = tail - ma
                self.window = list(self.baseline_scores[-self.depth:])
        self.raw_stream_scores = stream_scores
        
        self.logger.debug(f"  - Detrended baseline: {len(self.detrended_baseline)} points")
        if len(self.detrended_baseline) > 0:
            self.logger.debug(f"  - Detrended baseline range: [{self.detrended_baseline.min():.4f}, {self.detrended_baseline.max():.4f}]")
        
        self.spot.fit_baseline(self.detrended_baseline, np.array([]), init_quantile)
        self._baseline_fitted = True
        if not hasattr(self, 'window') or self.window is None:
            self.window = list(self.baseline_scores[-self.depth:]) if len(self.baseline_scores) >= self.depth else list(self.baseline_scores)
        self.logger.debug(f"  - Window initialized with last {min(self.depth, len(self.baseline_scores))} baseline points")

    def run_stream(self, update_with_alarm: bool = False, stream_data: np.ndarray = None):
        """
        Detrend stream on-the-fly, then delegate to SPOT.
        Only update the moving average window if status != Status.ALARM.
        """
        assert self._baseline_fitted, "Call fit_baseline() first."
        # Default to the stream provided at fit time if none is passed

        stream_data = self.raw_stream_scores if stream_data is None else stream_data
        
        alarms = []
        thresholds = []
        num_total = 0
        
        self.logger.debug(f"DriftSPOT.run_stream: Processing {len(stream_data)} points")
        
        for i, score in enumerate(stream_data):
            mean = np.mean(self.window)
            detrended_score = score - mean
            self.logger.debug(f"Point {i}: raw_score={score:.4f}, window_mean={mean:.4f}, detrended={detrended_score:.4f}")
            status = self.spot.evt.process(
                detrended_score,
                num_total + len(self.detrended_baseline),
                update_with_alarm=update_with_alarm,
            )
            current_threshold = self.spot.evt.get_current_threshold(
                num_total + len(self.detrended_baseline)
            )
            final_threshold = current_threshold + mean
            thresholds.append(final_threshold)
            self.logger.debug(f"  -> detrended_threshold={current_threshold:.4f}, final_threshold={final_threshold:.4f}")
            if status == Status.ALARM:
                alarms.append(i)
                # Don't update window on alarm to avoid skewing by anomalies
            else:
                num_total += 1
                self.window.pop(0)
                self.window.append(score)
            
        self.logger.debug("DriftSPOT.run_stream completed:")
        self.logger.debug(f"  - Total alarms: {len(alarms)}")
        self.logger.debug(f"  - Alarm indices: {alarms}")
        self.logger.debug(f"  - Thresholds computed: {len(thresholds)}")
        if thresholds:
            self.logger.debug(f"  - Threshold range: [{min(thresholds):.4f}, {max(thresholds):.4f}]")
        
        return {"alarms": alarms, "thresholds": thresholds}


