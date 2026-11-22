"""A tiny stand-in model with known answers for the lens checks, for tests and a no-GPU demo.

`demo()` is a six-layer model: whatever a unit writes at layer L reaches the last layer through a map that depends on
the context, `A_L + noise_c`, far from the identity early and the identity at the end; the last residual goes through
an RMS norm and the unembedding. So:

- the **logit lens** (read the write directly through the unembedding) is poor early and good late;
- the toy's **R-lens** is the average map `A_L`, like a fitted Jacobian lens: good at every layer, not perfect;
- steering is measured, not assumed: the write is added in each context, and the log-probability change goes
  through the norm and a softmax.

`demo_units()` gives a graph's worth of units with labels: one feature that writes " Austin" (labelled "Austin"), one
that writes " Paris" but is labelled "Texas cities" (the label check should flag it), and a few unlabelled ones.
`demo_edges()` gives a toy wiring catalog over neuron poles and a graph whose edges partly follow it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from circuitcheck.lens.edges import EdgeCatalog
from circuitcheck.lens.types import LensUnit, Pole

VOCAB = [
    " Austin", " Dallas", " Houston", " Texas", " Paris", " France", " London", " Berlin", " cat", " dog", " bird",
    " fish", " red", " blue", " green", " run", " walk", " jump", " one", " two", " three", " the", " and", ".",
]  # fmt: skip


@dataclass
class ToyLensModel:
    W_U: np.ndarray  # (vocab, d)
    maps: list[np.ndarray]  # A_L: layer L's output -> last layer, on average over contexts, (d, d)
    jitter: list[np.ndarray]  # per layer, each context's deviation from A_L, (C, d, d)
    writes: dict[str, list[np.ndarray]]  # kind -> per layer (n_units, d)
    contexts: np.ndarray  # (n_contexts, d) residuals at the last layer
    vocab: list[str] = field(default_factory=lambda: list(VOCAB))

    def write_vectors(self, units: Sequence[LensUnit]) -> np.ndarray:
        return np.stack([u.sign * self.writes[u.kind][u.layer][u.index] for u in units])

    def n_units(self, layer: int, kind: str) -> int:
        return int(self.writes[kind][layer].shape[0])

    def lenses(self) -> list[str]:
        return ["logit", "R"]

    def lens_logits(self, layer: int, vectors: np.ndarray, lens: str) -> np.ndarray:
        v = np.asarray(vectors, dtype=np.float64)
        if lens == "R":
            v = v @ self.maps[layer].T
        elif lens != "logit":
            raise ValueError(f"unknown lens {lens!r}")
        out: np.ndarray = v @ self.W_U.T
        return out

    def steer(self, layer: int, vectors: np.ndarray) -> np.ndarray:
        def lsm(x: np.ndarray) -> np.ndarray:
            m = x.max(-1, keepdims=True)
            out: np.ndarray = x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
            return out

        def norm(x: np.ndarray) -> np.ndarray:
            out: np.ndarray = x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
            return out

        base = lsm(norm(self.contexts) @ self.W_U.T)  # (C, V)
        maps = self.maps[layer][None] + self.jitter[layer]  # (C, d, d)
        moved = np.einsum("cij,nj->nci", maps, np.asarray(vectors, dtype=np.float64))  # (n, C, d)
        res: np.ndarray = (lsm(norm(moved + self.contexts[None]) @ self.W_U.T) - base[None]).mean(1)
        return res

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [self.vocab[i] for i in ids]

    def vocab_mask(self) -> np.ndarray | None:
        return None

    def describe(self) -> dict[str, str]:
        return {"model": "toy linear model (6 layers)", "lenses": "logit, R (the average map)", "contexts": "8 random"}


def demo(
    seed: int = 0, d: int = 16, n_layers: int = 6, n_features: int = 64, n_neurons: int = 32, n_contexts: int = 8
) -> ToyLensModel:
    rng = np.random.default_rng(seed)
    V = len(VOCAB)
    W_U = rng.normal(0, 1, (V, d))
    maps, jitter = [], []
    for layer in range(n_layers):
        t = layer / (n_layers - 1)
        R = rng.normal(0, 1 / np.sqrt(d), (d, d))
        maps.append((1 - t) * R * 2.0 + t * np.eye(d))
        jitter.append(rng.normal(0, 0.6 * (1 - t) + 0.1, (n_contexts, d, d)) / np.sqrt(d))
    writes = {
        "feature": [rng.normal(0, 1 / np.sqrt(d), (n_features, d)) for _ in range(n_layers)],
        "neuron": [rng.normal(0, 1 / np.sqrt(d), (n_neurons, d)) for _ in range(n_layers)],
    }
    last = n_layers - 1
    pinv = np.linalg.pinv(W_U)  # (d, V): a write that raises one token's logit
    for i, tok in enumerate((" Austin", " Paris")):
        e = np.zeros(V)
        e[VOCAB.index(tok)] = 1.0
        v = pinv @ (e - e.mean())
        v = v / np.linalg.norm(v) + 0.5 * writes["feature"][last][i]  # mostly one token, plus a little of the rest
        writes["feature"][last][i] = v / np.linalg.norm(v)
    contexts = rng.normal(0, 1.0, (n_contexts, d))
    return ToyLensModel(W_U=W_U, maps=maps, jitter=jitter, writes=writes, contexts=contexts)


def demo_units() -> list[LensUnit]:
    """A graph's worth of units: two labelled features at the last layer and unlabelled ones at layers 1, 3 and 5."""
    units = [
        LensUnit(layer=5, index=0, dose=4.0, label="Austin, the capital of Texas"),
        LensUnit(layer=5, index=1, dose=4.0, label="Texas cities"),
    ]
    units += [LensUnit(layer=layer, index=i, dose=4.0) for layer in (1, 3, 5) for i in range(2, 6)]
    return units


def demo_edges(seed: int = 0) -> tuple[dict[tuple[str, str], float], EdgeCatalog]:
    """A wiring catalog over 6 layers x 40 neurons (each pole wired to 6 later poles) and a 60-edge neuron graph: 24
    edges taken from the catalog (with its signs), 36 random forward edges."""
    rng = random.Random(seed)
    n_layers, n = 6, 40
    pairs: dict[tuple[Pole, Pole], float] = {}
    for layer in range(n_layers - 1):
        for i in range(n):
            for s in (1, -1):
                for _ in range(6):
                    m = rng.randrange(layer + 1, n_layers)
                    b: Pole = (m, rng.randrange(n), rng.choice((1, -1)))
                    pairs[((layer, i, s), b)] = rng.choice((1, -1)) * rng.uniform(0.2, 1.5)
    catalog = EdgeCatalog(pairs, range(n_layers), name="toy wiring", table="edges")
    graph: dict[tuple[str, str], float] = {}
    keys = list(pairs)
    while len(graph) < 24:
        a, b = rng.choice(keys)
        e = (f"{a[0]}:{a[1]}{'+' if a[2] > 0 else '-'}", f"{b[0]}:{b[1]}{'+' if b[2] > 0 else '-'}")
        graph[e] = pairs[(a, b)] * rng.uniform(0.5, 2.0)
    while len(graph) < 60:
        la = rng.randrange(n_layers - 1)
        lb = rng.randrange(la + 1, n_layers)
        e = (f"{la}:{rng.randrange(n)}{rng.choice('+-')}", f"{lb}:{rng.randrange(n)}{rng.choice('+-')}")
        graph.setdefault(e, rng.uniform(-1, 1))
    return graph, catalog
