"""Two-layer immune pattern recognizer for network traffic.

    Layer 1  - adaptive memory (B-cells, clonal selection):
               "Have I seen this *pattern* before?"  If a memory cell binds the
               record tightly enough, return its label (normal, DoS, Probe...).
    Layer 2  - self/non-self tolerance (T-cells, negative selection):
               For records memory does *not* recognise: "Is this at least
               *me*?"  V-Detector says self -> normal; non-self -> NOVEL.

Analogy: a hospital triage desk.  The receptionist (memory) instantly
recognises regular patients and frequent troublemakers by face.  Anyone she
does not recognise is sent to the security guard (negative selection), who
has only ever been told what staff look like - so he stops *anybody* who is
not staff, even a burglar nobody has ever seen.

Two fallback policies for unrecognised records are offered:
``"tolerance"`` (self -> normal; fewer alarms) and ``"alarm_union"``
(self -> memory's nearest label; negative selection can only add alarms).

NOVEL outputs are exactly the cases an analyst should label; feeding a few of
them back through :meth:`ImmuneRecognizer.learn_new_pattern` is the
"vaccination" step that creates new memory (secondary response).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clonal_selection import ClonalSelectionClassifier
from .negative_selection import VDetector

NOVEL = "novel"


@dataclass
class ImmuneRecognizer:
    vdetector: VDetector = field(default_factory=VDetector)
    clonal: ClonalSelectionClassifier = field(default_factory=ClonalSelectionClassifier)
    normal_label: str = "normal"
    policy: str = "tolerance"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ImmuneRecognizer":
        y = np.asarray(y, dtype=object)
        self.vdetector.fit(X[y == self.normal_label])
        self.clonal.fit(X, y)
        return self

    def explain(self, X: np.ndarray) -> dict:
        """Return every intermediate signal (useful for analysts and plots)."""
        lab, aff_dist, recognised = self.clonal.predict_with_affinity(X)
        nsa = self.vdetector.predict(X)
        if self.policy == "tolerance":
            # Unrecognised + tolerated by T-cells -> treat as normal.
            fallback = np.where(nsa == 1, NOVEL, self.normal_label)
        elif self.policy == "alarm_union":
            # Negative selection can only *add* alarms: unrecognised records
            # that look non-self are NOVEL, otherwise keep memory's best guess.
            fallback = np.where(nsa == 1, NOVEL, lab)
        else:
            raise ValueError(f"unknown policy {self.policy!r}")
        final = np.where(recognised, lab, fallback).astype(object)
        return {"final": final, "memory_label": lab, "memory_distance": aff_dist,
                "recognised": recognised, "nonself": nsa}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.explain(X)["final"]

    def is_attack(self, X: np.ndarray) -> np.ndarray:
        return (self.predict(X) != self.normal_label).astype(int)

    def learn_new_pattern(self, X_new: np.ndarray, label: str, n_memory: int | None = None):
        self.clonal.learn_new_pattern(X_new, label, n_memory)
        return self
