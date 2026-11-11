"""Clonal-selection pattern classifier (CLONALG-style) with immune memory.

Biology (Feynman version)
-------------------------
When a germ shows up, the few B-cells whose antibodies *roughly* fit it get
to multiply.  The better the fit, the more copies they make (clonal
expansion), and the copies are made deliberately sloppy - the better the
fit, the *less* sloppy (somatic hypermutation).  Copies that fit better
survive (affinity maturation) and some become long-lived *memory cells*, so
the next infection by the same germ is beaten faster: the secondary response.

Think of it as a locksmith with a box of keys (memory cells).  Shown a new
lock (antigen) he picks the 3 closest keys, cuts many slightly-different
copies of the best-fitting ones (few, careful cuts for good keys, many wild
cuts for poor ones) and keeps any copy that fits better than the original.

Algorithm (de Castro & Von Zuben, 2002, pattern-recognition variant)
--------------------------------------------------------------------
For every class c keep a memory population M_c.  Each generation, for each
training antigen x of class c:

1. **Selection** - choose the ``n_select`` cells of M_c with highest
   affinity (lowest Euclidean distance) to x.
2. **Clonal expansion** - cell of rank i gets ``round(clone_factor/(i+1))``
   clones (proportional to affinity rank).
3. **Hypermutation** - clone = cell + N(0, (mutation_scale * d_i)^2 / dim): the
   mutation step is proportional to distance, i.e. *inversely* proportional to
   affinity.
4. **Negative-selection operator** (Kim & Bentley, 2001) - discard clones
   that would be closer to another class's memory than to x (they would be
   "auto-reactive" against other patterns).
5. **Affinity maturation** - the best surviving clone (from the whole clone
   pool) replaces the best-matching memory cell if it binds x more tightly.

After each generation, **receptor editing** replaces the least-stimulated
cells of each class with fresh antigens ("bone-marrow" recruits).

Classification is by the nearest memory cell(s).  The distance to the nearest
cell is an *affinity* that tells us whether a pattern is actually recognised
or merely "closest of a bad bunch" - this is what lets the system say
"I don't know this pattern" (novelty) instead of guessing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d2 = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T
    return np.maximum(d2, 0.0)


@dataclass
class ClonalSelectionClassifier:
    n_memory_per_class: int = 40
    memory_allocation: str = "balanced"
    n_select: int = 3
    clone_factor: int = 10
    mutation_scale: float = 0.5
    n_generations: int = 8
    antigens_per_generation: int = 1500
    receptor_editing: float = 0.1
    negative_selection_operator: bool = True
    recognition_quantile: float = 0.99
    k_neighbors: int = 1
    random_state: int | None = 0

    memory_: np.ndarray = field(init=False, repr=False, default=None)
    memory_labels_: np.ndarray = field(init=False, repr=False, default=None)
    classes_: np.ndarray = field(init=False, repr=False, default=None)
    thresholds_: dict = field(init=False, repr=False, default_factory=dict)
    fit_seconds_: float = field(init=False, default=0.0)
    history_: list = field(init=False, repr=False, default_factory=list)

    # ------------------------------------------------------------ internals
    def _rng(self) -> np.random.Generator:
        if not hasattr(self, "_rng_obj"):
            self._rng_obj = np.random.default_rng(self.random_state)
        return self._rng_obj

    def _mature(self, mem: dict, Xc: dict, train_labels: list) -> None:
        """Run clonal selection generations on the classes in ``train_labels``."""
        rng = self._rng()
        for gen in range(self.n_generations):
            # Present a shuffled stream of antigens from the trainable classes.
            stream = []
            for c in train_labels:
                n = min(self.antigens_per_generation, len(Xc[c]))
                pick = rng.choice(len(Xc[c]), size=n, replace=False)
                stream.extend((c, i) for i in pick)
            order = rng.permutation(len(stream))
            stim = {c: np.zeros(len(mem[c])) for c in train_labels}
            # Snapshot of other-class memory for the negative-selection operator
            # (refreshed once per generation for speed).
            others_of = {
                c: (np.vstack([mem[o] for o in mem if o != c]) if len(mem) > 1 else None)
                for c in train_labels
            }
            improved = 0
            for j in order:
                c, i = stream[j]
                x = Xc[c][i]
                M = mem[c]
                d = np.sqrt(np.sum((M - x) ** 2, axis=1))
                sel = np.argsort(d)[: self.n_select]
                stim[c][sel[0]] += 1
                others = others_of[c] if self.negative_selection_operator else None
                # Clonal expansion + hypermutation of the n_select best cells.
                pool = []
                for rank, cell in enumerate(sel):
                    n_cl = max(1, int(round(self.clone_factor / (rank + 1))))
                    # Per-coordinate std chosen so the expected step length is
                    # mutation_scale * d (i.e. inversely proportional to affinity).
                    sigma = self.mutation_scale * d[cell] / np.sqrt(M.shape[1])
                    pool.append(M[cell] + rng.normal(0.0, sigma, size=(n_cl, M.shape[1])))
                clones = np.vstack(pool)
                dc = np.sqrt(np.sum((clones - x) ** 2, axis=1))
                if others is not None:
                    d_other = np.sqrt(_sqdist(clones, others).min(axis=1))
                    dc = np.where(d_other < dc, np.inf, dc)
                b = int(np.argmin(dc))
                # Affinity maturation: the best mature clone replaces the
                # best-matching memory cell only if it binds x more tightly
                # (one memory update per antigen, as in CLONALG).
                bmu = sel[0]
                if dc[b] < d[bmu]:
                    M[bmu] = clones[b]
                    improved += 1
            # Receptor editing: replace the least-stimulated cells.
            for c in train_labels:
                n_edit = int(self.receptor_editing * len(mem[c]))
                if n_edit and len(Xc[c]) > len(mem[c]):
                    worst = np.argsort(stim[c])[:n_edit]
                    worst = worst[stim[c][worst] == 0]
                    if len(worst):
                        mem[c][worst] = Xc[c][rng.choice(len(Xc[c]), size=len(worst))]
            self.history_.append({"generation": len(self.history_), "improved": improved})

    def _memory_size(self, n_c: int, sizes: dict) -> int:
        """Memory cells for a class of n_c antigens.

        * ``balanced`` - the same budget for every class (favours rare classes).
        * ``sqrt`` - budget scales with sqrt(n_c / median class size), so large,
          diverse classes such as normal traffic get more cells (fewer false
          alarms) while rare classes keep a useful minimum.
        """
        if self.memory_allocation == "balanced":
            return self.n_memory_per_class
        if self.memory_allocation == "sqrt":
            med = float(np.median(list(sizes.values())))
            return int(np.clip(round(self.n_memory_per_class * np.sqrt(n_c / med)),
                               10, 8 * self.n_memory_per_class))
        raise ValueError(f"unknown memory_allocation {self.memory_allocation!r}")

    def _pack(self, mem: dict) -> None:
        self.memory_ = np.vstack([mem[c] for c in self.classes_])
        self.memory_labels_ = np.concatenate(
            [np.full(len(mem[c]), c, dtype=object) for c in self.classes_])

    def _calibrate(self, Xc: dict, labels: list, max_samples: int = 3000) -> None:
        """Recognition threshold ("activation threshold") per class.

        tau_c = max( q-quantile of antigen -> nearest own memory cell,
                     q-quantile of antigen -> nearest *other* antigen of c ).
        The second term is a leave-one-out spacing estimate; without it a
        class learned from a handful of examples (whose memory cells sit on
        the examples themselves) would get tau = 0 and never be recognised.
        """
        rng = np.random.default_rng(0)
        for c in labels:
            Xs = Xc[c]
            if len(Xs) > max_samples:
                Xs = Xs[rng.choice(len(Xs), max_samples, replace=False)]
            own = self.memory_[self.memory_labels_ == c]
            d_mem = np.sqrt(_sqdist(Xs, own).min(axis=1))
            tau = float(np.quantile(d_mem, self.recognition_quantile))
            if len(Xs) > 1:
                d2 = _sqdist(Xs, Xs)
                np.fill_diagonal(d2, np.inf)
                d_loo = np.sqrt(d2.min(axis=1))
                tau = max(tau, float(np.quantile(d_loo, self.recognition_quantile)))
            self.thresholds_[c] = tau

    # ----------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClonalSelectionClassifier":
        t0 = time.perf_counter()
        X = np.asarray(X, float)
        y = np.asarray(y, dtype=object)
        rng = self._rng()
        self.classes_ = np.array(sorted(set(y)), dtype=object)
        Xc = {c: X[y == c] for c in self.classes_}
        mem = {}
        sizes = {c: len(Xc[c]) for c in self.classes_}
        for c in self.classes_:
            m = min(self._memory_size(sizes[c], sizes), len(Xc[c]))
            mem[c] = Xc[c][rng.choice(len(Xc[c]), size=m, replace=False)].copy()
        self.history_ = []
        self._mature(mem, Xc, list(self.classes_))
        self._mem = mem
        self._pack(mem)
        self._calibrate(Xc, list(self.classes_))
        self.fit_seconds_ = time.perf_counter() - t0
        return self

    def learn_new_pattern(self, X_new: np.ndarray, label: str,
                          n_memory: int | None = None) -> "ClonalSelectionClassifier":
        """Secondary-response style incremental learning of a new class.

        Existing memory is frozen; only the new class's memory cells are
        matured (against the others via the negative-selection operator).
        No retraining on old data is needed.
        """
        X_new = np.asarray(X_new, float)
        rng = self._rng()
        m = min(n_memory or self.n_memory_per_class, len(X_new))
        mem = dict(self._mem)
        mem[label] = X_new[rng.choice(len(X_new), size=m, replace=False)].copy()
        if label not in set(self.classes_):
            self.classes_ = np.array(list(self.classes_) + [label], dtype=object)
        self._mature(mem, {label: X_new}, [label])
        self._mem = mem
        self._pack(mem)
        self._calibrate({label: X_new}, [label])
        return self

    def kneighbors_memory(self, X: np.ndarray, chunk: int = 8192):
        X = np.asarray(X, float)
        k = self.k_neighbors
        idx = np.empty((len(X), k), dtype=int)
        dist = np.empty((len(X), k))
        for s in range(0, len(X), chunk):
            d2 = _sqdist(X[s:s + chunk], self.memory_)
            part = np.argpartition(d2, kth=min(k, d2.shape[1]) - 1, axis=1)[:, :k]
            pd_ = np.take_along_axis(d2, part, axis=1)
            o = np.argsort(pd_, axis=1)
            idx[s:s + chunk] = np.take_along_axis(part, o, axis=1)
            dist[s:s + chunk] = np.sqrt(np.take_along_axis(pd_, o, axis=1))
        return dist, idx

    def predict_with_affinity(self, X: np.ndarray):
        """Return (labels, distance to nearest memory cell, recognised mask)."""
        dist, idx = self.kneighbors_memory(X)
        labs = self.memory_labels_[idx]
        if self.k_neighbors == 1:
            pred = labs[:, 0]
        else:
            w = 1.0 / (dist + 1e-9)
            pred = np.empty(len(X), dtype=object)
            for i in range(len(X)):
                score: dict = {}
                for l, wi in zip(labs[i], w[i]):
                    score[l] = score.get(l, 0.0) + wi
                pred[i] = max(score, key=score.get)
        d_near = dist[:, 0]
        thr = np.array([self.thresholds_[p] for p in pred])
        return pred, d_near, d_near <= thr

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_affinity(X)[0]

    @property
    def n_memory_cells(self) -> int:
        return 0 if self.memory_ is None else len(self.memory_)
