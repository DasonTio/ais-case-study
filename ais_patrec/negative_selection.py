"""Real-valued Negative Selection with variable-sized detectors (V-Detector).

Biology (Feynman version)
-------------------------
Your thymus is a strict boot camp.  It manufactures T-cells with *random*
receptors, then shows every recruit samples of your own body ("self").  Any
recruit that reacts to self is executed.  The survivors can only ever react
to things that are *not you* - so the body gets an "anything foreign" alarm
without ever having seen the foreign thing.  (Forrest et al., 1994.)

Algorithm (Ji & Dasgupta, 2004; 2009)
-------------------------------------
* Self = normal traffic, points in [0, 1]^d, each with a tolerance radius r_s.
* Repeatedly draw a random candidate detector centre x.
* If x lies inside an existing detector it is "already covered"; count it.
  When ``1 / (1 - c0)`` consecutive candidates are already covered we
  estimate that at least a fraction c0 of non-self space is covered and stop.
* Otherwise give x the *largest radius that does not touch self*:
  r = dist(x, nearest self) - r_s.  If r > 0 the detector "matures".
* A sample is non-self (anomalous) if it falls inside any detector.

Known weaknesses (kept visible on purpose)
------------------------------------------
* Coverage "holes" near the self/non-self boundary (Li et al., 2021).
* Random generation scales poorly with dimension (Stibor et al., 2005), which
  is why :class:`~ais_patrec.data.ShapeSpace` projects to a low-dim space.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class VDetector:
    """V-Detector negative selection anomaly detector.

    Parameters
    ----------
    self_radius:
        Tolerance radius r_s around each self sample (shape-space units).
    target_coverage:
        Estimated non-self coverage c0 at which generation stops.
    max_detectors:
        Hard cap on mature detectors.
    max_candidates:
        Hard cap on random candidates drawn (guards against non-termination,
        which Stibor et al. (2005) showed is sensitive to parameters).
    low, high:
        Bounds of the shape-space hyper-rectangle in which candidates are drawn.
    """

    self_radius: float = 0.05
    target_coverage: float = 0.99
    max_detectors: int = 3000
    max_candidates: int = 300_000
    low: float = 0.0
    high: float = 1.0
    candidate_sampler: str = "uniform"
    proximity_sigma: float = 0.1
    batch_size: int = 4096
    random_state: int | None = 0

    centers_: np.ndarray = field(init=False, repr=False, default=None)
    radii_: np.ndarray = field(init=False, repr=False, default=None)
    n_candidates_: int = field(init=False, default=0)
    stop_reason_: str = field(init=False, default="")
    fit_seconds_: float = field(init=False, default=0.0)

    # ------------------------------------------------------------------ fit
    def fit(self, X_self: np.ndarray) -> "VDetector":
        """Train on self (normal) samples only."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_state)
        X_self = np.asarray(X_self, dtype=float)
        d = X_self.shape[1]
        self._self_nn = NearestNeighbors(n_neighbors=1).fit(X_self)

        centers = np.empty((self.max_detectors, d))
        radii = np.empty(self.max_detectors)
        n_det = 0
        covered_streak = 0
        streak_needed = int(np.ceil(1.0 / (1.0 - self.target_coverage)))
        n_cand = 0
        self.stop_reason_ = "max_candidates"

        while n_cand < self.max_candidates:
            cand = self._sample_candidates(rng, X_self, d)
            # Distance from each candidate to its nearest self sample (thymus test).
            d_self = self._self_nn.kneighbors(cand, return_distance=True)[0][:, 0]
            done = False
            for x, ds in zip(cand, d_self):
                n_cand += 1
                if n_det:
                    inside = np.any(
                        np.sum((centers[:n_det] - x) ** 2, axis=1) < radii[:n_det] ** 2
                    )
                else:
                    inside = False
                if inside:
                    covered_streak += 1
                    if covered_streak >= streak_needed:
                        self.stop_reason_ = "coverage_reached"
                        done = True
                        break
                    continue
                covered_streak = 0
                r = ds - self.self_radius
                if r > 0:  # survives negative selection -> mature detector
                    centers[n_det] = x
                    radii[n_det] = r
                    n_det += 1
                    if n_det >= self.max_detectors:
                        self.stop_reason_ = "max_detectors"
                        done = True
                        break
                if n_cand >= self.max_candidates:
                    break
            if done:
                break

        self.centers_ = centers[:n_det].copy()
        self.radii_ = radii[:n_det].copy()
        self.n_candidates_ = n_cand
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def _sample_candidates(self, rng, X_self, d) -> np.ndarray:
        """Draw random receptor centres.

        * ``"uniform"`` - classic V-Detector: uniform over the hyper-rectangle.
        * ``"self_proximal"`` - centres drawn around randomly chosen self samples
          (x = s + N(0, sigma^2 I)).  This follows the density-guided family of
          NSAs (e.g. Yang, Chen & Li 2017, ASD-RNSA / RNSAP) that place detectors
          where antigens actually live, so the coverage estimate measures the
          region bordering self instead of empty corners of the cube.
        """
        if self.candidate_sampler == "uniform":
            return rng.uniform(self.low, self.high, size=(self.batch_size, d))
        if self.candidate_sampler == "self_proximal":
            base = X_self[rng.integers(0, len(X_self), size=self.batch_size)]
            return base + rng.normal(0.0, self.proximity_sigma, size=base.shape)
        raise ValueError(f"unknown candidate_sampler {self.candidate_sampler!r}")

    # -------------------------------------------------------------- predict
    def _margin(self, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """max_j (r_j - ||x - c_j||): > 0 means x is inside some detector."""
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X))
        c2 = np.sum(self.centers_ ** 2, axis=1)
        for s in range(0, len(X), chunk):
            xb = X[s:s + chunk]
            d2 = np.sum(xb ** 2, axis=1)[:, None] + c2[None, :] - 2 * xb @ self.centers_.T
            dist = np.sqrt(np.maximum(d2, 0.0))
            out[s:s + chunk] = np.max(self.radii_[None, :] - dist, axis=1)
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Continuous anomaly score (higher = more non-self)."""
        if len(self.radii_) == 0:
            return np.full(len(X), -np.inf)
        return self._margin(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = non-self (anomaly), 0 = self (normal)."""
        return (self.decision_function(X) > 0).astype(int)

    @property
    def n_detectors(self) -> int:
        return 0 if self.radii_ is None else len(self.radii_)


@dataclass
class NearestSelfDetector:
    """'Self-detector' / positive-selection baseline (Stibor et al., 2005).

    Flags x if its distance to the nearest self sample exceeds ``self_radius``.
    Stibor et al. argued this trivially simple rule matches or beats negative
    selection - included so the comparison is honest.
    """

    self_radius: float = 0.05

    def fit(self, X_self: np.ndarray) -> "NearestSelfDetector":
        self._nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(X_self, float))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._nn.kneighbors(X)[0][:, 0] - self.self_radius

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)
