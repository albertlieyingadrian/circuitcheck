"""Diff a model before and after fine-tuning in the neuron basis, with circuitcheck's paired swap null.

Fine-tuning updates the same weights, so neuron `L12/N345` exists in both models and can be compared directly: no
dictionary is needed. For each prompt and each model, every neuron gets one number:

- `tail_rate` (default): the share of the prompt's tokens where the neuron fires in its tail, beyond the *before*
  model's p`q` or below its p(100 - `q`). The thresholds come from the before model (on `reference` texts, or the
  prompts themselves) and are used for both models, so a change in rate is a change in behaviour on a fixed scale.
  A neuron counts as present on a prompt when it fires in its tail there at least once;
- `mean`: the neuron's mean activation over the prompt's tokens (present everywhere; a signed shift).

Each prompt becomes a `GraphSummary` with those numbers as its `features` (units `"layer:neuron"`), and
`circuitcheck.diff.diff_graphs` does the rest: gained / lost / up / down statuses, the paired swap null with a
Benjamini-Hochberg correction, bootstrap intervals, the split-half check and the optional control contrast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from circuitcheck.diff.core import DiffReport, Settings, Validity, diff_graphs
from circuitcheck.diff.graphs import GraphSummary, unit

Stat = Literal["tail_rate", "mean"]
Thresholds = dict[int, tuple[np.ndarray, np.ndarray]]

NEURON_CAVEAT = (
    "This diff is in the model's own MLP-neuron basis: there is no transcoder, so no dictionary trained on the base "
    "model can go stale. Two things still limit it: the tail thresholds come from the model before training, and a "
    "neuron that changed rate may have changed meaning too. Treat gained and lost neurons as candidates to describe "
    "(circuitcheck.neurons.label_neuron) and test (circuitcheck.validate with a NeuronBackend), not as findings."
)


def _acts(model: Any, tokenizer: Any, text: str, layers: Sequence[int], max_len: int) -> dict[int, np.ndarray]:
    """(tokens, neurons) activations per layer on one text."""
    import neuronkit as nk
    import torch

    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_len]
    dev = next(model.parameters()).device
    with nk.capture(model, layers=list(layers)) as store, torch.no_grad():
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    return {layer: store[layer][0][0].numpy() for layer in layers}


def _layers(model: Any, layers: Sequence[int] | None) -> list[int]:
    import neuronkit as nk

    return list(layers) if layers is not None else list(range(len(nk.sites(model))))


def tail_thresholds(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    q: float = 99.0,
    max_len: int = 256,
) -> Thresholds:
    """Per-neuron (low, high) tail thresholds: the p(100 - q) and p(q) of each neuron's activations over `texts`."""
    lay = _layers(model, layers)
    per: dict[int, list[np.ndarray]] = {layer: [] for layer in lay}
    for t in texts:
        for layer, a in _acts(model, tokenizer, t, lay, max_len).items():
            per[layer].append(a)
    out: Thresholds = {}
    for layer, chunks in per.items():
        x = np.concatenate(chunks, axis=0)
        out[layer] = (np.nanpercentile(x, 100 - q, axis=0), np.nanpercentile(x, q, axis=0))
    return out


def neuron_summaries(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    thresholds: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    stat: Stat = "tail_rate",
    layers: Sequence[int] | None = None,
    name: str = "",
    max_len: int = 256,
) -> list[GraphSummary]:
    """One `GraphSummary` per prompt whose `features` are per-neuron tail rates (or mean activations)."""
    if stat == "tail_rate" and thresholds is None:
        raise ValueError("tail_rate needs thresholds: see tail_thresholds()")
    lay = sorted(thresholds) if thresholds is not None and layers is None else _layers(model, layers)
    out = []
    for i, p in enumerate(prompts):
        feats: dict[str, float] = {}
        for layer, a in _acts(model, tokenizer, p, lay, max_len).items():
            if stat == "tail_rate":
                assert thresholds is not None
                lo, hi = thresholds[layer]
                tail = ((a > hi[None, :]) | (a < lo[None, :])).astype(float)
                tail[np.isnan(a)] = np.nan
                val = np.nanmean(tail, axis=0)
            else:
                val = np.nanmean(a, axis=0)
            for idx in np.nonzero(np.nan_to_num(val) != 0)[0]:
                feats[unit(layer, int(idx))] = float(val[idx])
        out.append(GraphSummary(name=f"{name}{i}", prompt=p, features=feats, meta={"basis": "neuron", "stat": stat}))
    return out


def diff_neurons(
    before: Any,
    after: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    stat: Stat = "tail_rate",
    q: float = 99.0,
    reference: Sequence[str] | None = None,
    layers: Sequence[int] | None = None,
    control: Any | None = None,
    settings: Settings | None = None,
    labels: Mapping[str, str] | None = None,
    before_name: str = "before",
    after_name: str = "after",
    max_len: int = 256,
) -> DiffReport:
    """Which neurons did training make fire more (or less) on `prompts`? The same prompts run through both models,
    paired by index.

    `control` is a model from a training run that did not learn the behaviour (same base): a neuron's change is
    specific if the control run changed it much less.
    """
    th = tail_thresholds(before, tokenizer, reference or prompts, layers=layers, q=q, max_len=max_len)
    kw: dict[str, Any] = {"thresholds": th, "stat": stat, "layers": layers, "max_len": max_len}
    b = neuron_summaries(before, tokenizer, prompts, name="before", **kw)
    a = neuron_summaries(after, tokenizer, prompts, name="after", **kw)
    ctrl = (b, neuron_summaries(control, tokenizer, prompts, name="control", **kw)) if control is not None else None
    rep = diff_graphs(
        b, a, settings=settings, control=ctrl, labels=labels, before_name=before_name, after_name=after_name
    )
    what = f"tail rate beyond the before model's p{q:g} or p{100 - q:g}" if stat == "tail_rate" else "mean activation"
    return rep.model_copy(
        update={
            "basis": "neuron",
            "validity": Validity(ok=True, warnings=[]),
            "caveats": [f"Each neuron's number is its {what} on a prompt. {NEURON_CAVEAT}"],
        }
    )
