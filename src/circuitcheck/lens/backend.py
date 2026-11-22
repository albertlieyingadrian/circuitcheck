"""What a model must offer for the lens checks. circuit-tracer's ReplacementModel is wrapped in `tracer.py`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from circuitcheck.lens.types import LensUnit


class SteeringBackend(Protocol):
    """The calls `lens_predicts_steering` needs. Everything crosses this boundary as NumPy or plain Python.

    A unit's *write* is the vector one unit of its activation adds to the residual stream at its layer's output
    (a transcoder feature's decoder row, or `sign * W_out[n]` for a neuron pole).
    """

    def write_vectors(self, units: Sequence[LensUnit]) -> np.ndarray:
        """Each unit's write, shape (n, d_model), with the unit's sign applied."""
        ...

    def n_units(self, layer: int, kind: str) -> int:
        """How many units of this kind the layer has (random controls are drawn from range(n_units))."""
        ...

    def lens_logits(self, layer: int, vectors: np.ndarray, lens: str) -> np.ndarray:
        """A lens's next-token logits for each vector read at `layer`'s output, shape (n, vocab). No forward pass."""
        ...

    def steer(self, layer: int, vectors: np.ndarray) -> np.ndarray:
        """Mean change in next-token log-probabilities when each vector is added at `layer`'s output, at every
        position of the backend's contexts: shape (n, vocab)."""
        ...

    def lenses(self) -> list[str]:
        """Lens names `lens_logits` accepts, e.g. ['logit', 'R']."""
        ...

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        """Readable strings for token ids."""
        ...

    def vocab_mask(self) -> np.ndarray | None:
        """Tokens to keep in the correlations (bool, shape (vocab,)), or None for all."""
        ...

    def describe(self) -> dict[str, str]:
        """A few strings for the report: model, transcoders, contexts, device."""
        ...
