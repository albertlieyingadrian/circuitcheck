"""Neuron labels that are tested, not only written: describe a neuron pole, then check that the description holds.

The pipeline, per pole (the positive or negative end of one neuron), after "Neurons as an Interpretable Basis"
(https://www.goodfire.com/research/neurons-as-an-interpretable-basis) and run with neuronkit's `score_description`:

1. **Examples.** The corpus is run once; the pole's strongest firings (at most `per_doc` per document) become
   excerpts. The top `n_explain` are shown to the explainer; the next `n_positive` are held out.
2. **Describe** (unless a description is given): the explainer sees the excerpts with the firing tokens marked.
3. **Predict**: a judge gets the description and a shuffled mix of the held-out firing excerpts and random windows
   of the corpus, and says which ones fire. Scores: balanced accuracy and AUROC.
4. **Simulate**: the judge predicts a 0-10 activation for every token of the held-out excerpts. Those predictions,
   scaled to the pole's p99.9 magnitude, are patched in for the real activations, and compared with zero-ablating
   the neuron on the same tokens: recovered = 1 - KL(base || simulated) / KL(base || ablated). 1.0 means the
   description carries everything the neuron's activations carry there; 0.0 means no better than deleting it.
5. **Write** (optional, `n_write`): the judge writes texts meant to fire the pole; does one beat the corpus maximum?

The trust score is `recovered`: it is measured in the model's own output, so a description that only sounds right
does not earn it. Prediction accuracy and AUROC are reported next to it, in the field circuitcheck's feature labels
use (`detection`), so `render_labels` reads neuron labels unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from circuitcheck.labels.autointerp import DetectionScore, FeatureLabel
from circuitcheck.labels.providers import LLM


class NeuronLabel(FeatureLabel):
    """A `FeatureLabel` for one neuron pole, with the predict / simulate / write scores of its description."""

    layer: int
    neuron: int
    sign: int = Field(description="+1 for the positive pole, -1 for the negative pole")
    predict_accuracy: float | None = Field(None, description="balanced accuracy, firing excerpts vs random text")
    predict_auroc: float | None = None
    recovered: float | None = Field(None, description="share of the ablation damage the simulated activations undo")
    write_best: float | None = Field(None, description="strongest firing on the judge's written texts")
    write_beats_reference: bool | None = None
    reference_max: float | None = Field(None, description="the pole's strongest firing on the corpus")
    a_ref: float | None = Field(None, description="the pole's p99.9 magnitude, the scale simulated levels map onto")
    ablation_kl: float | None = Field(
        None,
        description="mean KL(base || ablated) over the simulated windows; near 0 means the neuron does nothing there, "
        "so there is no damage to recover and `recovered` is None",
    )
    n_windows: int = Field(0, description="held-out windows whose simulation was patched in")
    unparsed: int = Field(0, description="judge replies that could not be parsed")

    @property
    def score(self) -> float | None:
        """The trust score: the share of ablation damage recovered by simulating from the description."""
        return self.recovered


def neuron_unit(layer: int, index: int, sign: int = 1) -> str:
    """'layer:nINDEX+' (or '-'): the neuron-pole syntax the `lens` command also reads."""
    return f"{layer}:n{index}{'+' if sign > 0 else '-'}"


def parse_pole(spec: str) -> tuple[int, int, int]:
    """'12:n345+', '12:345-' or 'L12/N345' (positive) -> (layer, neuron, sign)."""
    s = spec.strip()
    sign = -1 if s.endswith("-") else 1
    s = s.rstrip("+-").replace("L", "").replace("N", "").replace("n", "").replace("/", ":")
    layer, index = s.split(":")
    return int(layer), int(index), sign


def _finite(x: float | None) -> float | None:
    return None if x is None or math.isnan(x) else float(x)


def label_neuron(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layer: int,
    index: int,
    sign: int = 1,
    *,
    llm: LLM,
    judge: LLM | None = None,
    description: str | None = None,
    n_explain: int = 12,
    n_positive: int = 8,
    n_negative: int = 8,
    n_simulate: int = 6,
    n_write: int = 0,
    per_doc: int = 2,
    window: int = 32,
    max_len: int = 256,
    seed: int = 0,
) -> NeuronLabel:
    """Describe one neuron pole from `texts` (or take `description` as given) and test the description.

    `llm` writes the description; `judge` (default: the same model) predicts and simulates. The excerpts the
    explainer saw are never scored.
    """
    import neuronkit as nk
    import torch
    from neuronkit import describe as nkd
    from neuronkit.examples import excerpt

    unit = neuron_unit(layer, index, sign)
    neuron = nk.Neuron(layer, index)

    def failed(msg: str) -> NeuronLabel:
        return NeuronLabel(
            unit=unit, layer=layer, neuron=index, sign=sign, label=description or "", model=llm.model, error=msg
        )

    try:
        acts, toks = nk.run_capture(model, nk.batches(tokenizer, list(texts), max_len), [neuron])
        per = acts[neuron]
        pole = torch.cat([torch.nan_to_num(sign * a.float(), nan=0.0) for a in per]).clamp(min=0)
        if not pole.numel() or float(pole.max()) <= 0:
            return failed("the pole never fires on these texts")
        a_ref = float(torch.quantile(pole, 0.999))
        reference_max = float(pole.max())
        fir = nk.top_firings(per, n_explain + n_positive, sign, per_doc)
        ex = [excerpt(tokenizer, toks[f.doc], per[f.doc], f.pos, doc=f.doc) for f in fir]
        shown, held = ex[:n_explain], ex[n_explain:]
        if not held:
            return failed(f"only {len(shown)} firings, none left to hold out: lower n_explain or add texts")
        if description is None:
            system, prompt = nkd.describe_prompt(shown, sign)
            description = nkd.parse_description(llm.complete(system, prompt, 300))
        negatives = nk.random_windows(tokenizer, toks, per, n_negative, width=window, seed=seed)
        sim = [(toks[e.doc], e) for e in held[:n_simulate]]
        sc = nkd.score_description(
            model,
            tokenizer,
            neuron,
            sign,
            description,
            judge or llm,
            held,
            negatives,
            sim,
            a_ref,
            reference_max,
            n_write=n_write,
            seed=seed,
        )
    except Exception as e:  # one bad neuron or reply should not sink a batch; the error is kept on the record
        return failed(f"{type(e).__name__}: {e}")
    acc, auc = _finite(sc.predict_accuracy), _finite(sc.predict_auroc)
    detection = None
    if acc is not None:
        detection = DetectionScore(
            auroc=auc,
            balanced_accuracy=acc,
            brier=float("nan"),
            n_positive=len(held),
            n_negative=len(negatives),
            probs=[],
            truth=[],
        )
    return NeuronLabel(
        unit=unit,
        layer=layer,
        neuron=index,
        sign=sign,
        label=description,
        model=llm.model,
        detection=detection,
        n_explain=len(shown),
        predict_accuracy=acc,
        predict_auroc=auc,
        recovered=_finite(sc.recovered),
        write_best=sc.write_best,
        write_beats_reference=sc.write_beats_reference,
        reference_max=reference_max,
        a_ref=a_ref,
        ablation_kl=sum(w.ablate for w in sc.windows) / len(sc.windows) if sc.windows else None,
        n_windows=len(sc.windows),
        unparsed=sc.unparsed,
    )


def label_neurons(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    poles: Sequence[str | tuple[int, int, int]],
    *,
    llm: LLM,
    descriptions: Mapping[str, str] | None = None,
    **kw: Any,
) -> list[NeuronLabel]:
    """`label_neuron` for several poles ('12:n345+' or (layer, neuron, sign)). `descriptions` maps a pole's unit
    string to a description to test instead of writing one (for example, a description from another tool)."""
    out = []
    for p in poles:
        layer, index, sign = parse_pole(p) if isinstance(p, str) else p
        given = (descriptions or {}).get(neuron_unit(layer, index, sign))
        out.append(label_neuron(model, tokenizer, texts, layer, index, sign, llm=llm, description=given, **kw))
    return out
