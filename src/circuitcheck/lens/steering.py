"""Does a unit's lens read predict what steering it does? Rank agreement against random units of the same layer.

Adapted from neuron-lens's causal check (github.com/stdlogits/neuron-lens, MIT: `scripts/checks/causal_lens.py`,
docs/findings.md §3 "A write's lens read predicts what steering it does"). For every unit under test (graph features
or neuron poles):

1. **Write.** The vector one unit of activation adds to the residual stream at its layer's output.
2. **Lens read.** A lens's next-token logits for that vector, from the weights alone: the logit lens (the final norm
   and unembedding) or a Jacobian lens such as neuron-lens's R-lens, which first carries the vector to the last
   layer.
3. **Steering.** The measured mean change in next-token log-probabilities when `dose x write` is added at the layer's
   output, at every position of a few neutral contexts.
4. **Agreement.** Spearman rho (rank agreement) between 2 and 3 over the vocabulary, plus Pearson r and top-k overlap.
5. **Controls.** `n_random` random units of the same layer, each pushed with the same write norm as a tested unit,
   go through 1-4. A *mismatched* pair (another unit's read against this unit's steering) gives the chance level.
6. **Labels.** A unit's label agrees with a read when one of its content words names one of the read's top tokens. A
   unit whose label agrees with neither the lens reads nor the steering is flagged.

The rules behind every flag and verdict are in `RULES` and printed in each report.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np

from circuitcheck.lens.backend import SteeringBackend
from circuitcheck.lens.stats import bootstrap, pearson_rows, spearman_rows, topk_overlap
from circuitcheck.lens.types import (
    Interval,
    LabelCheck,
    LayerAgreement,
    LensReport,
    LensSummary,
    LensUnit,
    UnitResult,
    UnitScore,
)
from circuitcheck.validate.types import FeatureNode

LabelJudge = Callable[[str, Sequence[str]], bool]

LOW_Q = 0.05  # a tested unit whose rho is below this quantile of its layer's random units is flagged ...
MISS_MARGIN = 0.1  # ... and at least this far below their median
NAN = float("nan")

RULES = [
    "rho: Spearman rank correlation, over the kept vocabulary, between a lens's logits for the unit's write and the "
    "mean change in log-probabilities when the write is added at the unit's layer (dose x write, every position)",
    "random units: drawn from the same layer and kind, each pushed with the same write norm as a tested unit",
    "chance: another unit's read (same layer) against this unit's steering (the mismatched pair)",
    "per layer and overall: median rho of tested units minus that of random units, 95% bootstrap interval "
    "(tested and random units resampled separately); overall weights each layer by its tested units",
    "verdict: more (or less) lens-predictable than random units when the overall interval excludes 0",
    f"flag 'lens misses': rho below the {LOW_Q:.0%} quantile of its layer's random units (10+ random units) and at "
    f"least {MISS_MARGIN} below their median",
    "flag 'label disagrees': the label names none of the top tokens of any lens read nor of the steering",
]

_STOP_WORDS = (
    "the and for with that this from into onto over under about after before when where which while what who whom "
    "its it's their them they then than also only such some any all each other more most less very can may might "
    "will would should could not but are was were been being has have had does did doing word words token tokens "
    "context contexts text texts like etc mostly often usually esp especially related things thing something "
    "around near following preceding writes write fires fire firing promotes promoting suppresses boosts boosting "
    "predicts predicting output outputs mixed various general"
)
_STOP = frozenset(_STOP_WORDS.split())


def _norm_token(t: str) -> str:
    t = t.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n").strip().lower()
    return t.strip(".,;:!?\"'`()[]{}")


def label_words(label: str) -> list[str]:
    """Content words of a label (lower case, 3+ letters, stop words dropped) and its quoted strings."""
    quoted = [q.strip().lower() for q in re.findall(r"['\"“‘`]([^'\"”’`]{1,24})['\"”’`]", label)]
    words = [w.lower() for w in re.findall(r"[^\W\d_][\w'-]*", label)]
    out: list[str] = []
    for w in [*quoted, *words]:
        w = w.strip("'-")
        if w and w not in out and (len(w) >= 3 or not w.isascii()) and w not in _STOP:
            out.append(w)
    return out


# Words that name punctuation or layout, and the characters they name ("" means a whitespace-only token).
SYMBOL_WORDS: dict[str, tuple[str, ...]] = {
    "semicolon": (";",),
    "colon": (":", "："),
    "comma": (",", "，"),
    "period": (".", "。"),
    "full stop": (".", "。"),
    "ellipsis": ("...", "…"),
    "quote": ('"', "'", "“", "”", "‘", "’"),
    "quotation": ('"', "'", "“", "”"),
    "apostrophe": ("'", "’"),
    "newline": ("\n",),
    "line break": ("\n",),
    "asterisk": ("*",),
    "bold": ("**",),
    "bracket": ("(", ")", "[", "]", "{", "}"),
    "parenthes": ("(", ")"),
    "paren": ("(", ")"),
    "brace": ("{", "}"),
    "hyphen": ("-",),
    "dash": ("-", "—", "–"),
    "slash": ("/",),
    "backslash": ("\\",),
    "question mark": ("?", "？"),
    "exclamation": ("!", "！"),
    "hash": ("#",),
    "underscore": ("_",),
    "space": ("",),
    "whitespace": ("",),
    "indentation": ("",),
    "tab": ("\t",),
}


def _names_symbol(label: str, raw: Sequence[str]) -> bool:
    low = label.lower()
    for name, chars in SYMBOL_WORDS.items():
        if re.search(rf"\b{name}", low):
            for t in raw:
                if chars == ("",):
                    if t and not t.strip():
                        return True
                elif any(c in t for c in chars):
                    return True
    if re.search(r"\bpunctuation", low):
        return any(t.strip() and not any(ch.isalnum() for ch in t) for t in raw)
    return False


def label_matches(label: str, tokens: Sequence[str]) -> bool:
    """The default label judge: does a content word of the label name one of the tokens?

    A word names a token when they are equal, or one starts with the other and both have 3+ characters
    (`animal` names ` animals`; `pets` names ` pet`). Words for punctuation and layout name the characters
    (`semicolon` names `;`, `spaces` a whitespace-only token; `SYMBOL_WORDS`), and `punctuation` names any token
    without letters or digits. It is a word match, not a judgement of meaning: pass `label_judge` for that.
    """
    toks = [t for t in (_norm_token(x) for x in tokens) if t]
    for w in label_words(label):
        for t in toks:
            if w == t or (len(t) >= 3 and len(w) >= 3 and (w.startswith(t) or t.startswith(w))):
                return True
    return _names_symbol(label, [str(x) for x in tokens])


def units_from_nodes(nodes: Sequence[FeatureNode], dose: str = "activation") -> list[LensUnit]:
    """Graph feature nodes as lens units, one per (layer, feature): the largest activation over its positions is
    the dose (`dose="activation"`), or 1 (`dose="unit"`). Labels are kept."""
    best: dict[tuple[int, int], FeatureNode] = {}
    for n in nodes:
        k = (n.layer, n.feature)
        if k not in best or abs(n.activation) > abs(best[k].activation):
            best[k] = n
    return [
        LensUnit(
            layer=n.layer,
            index=n.feature,
            kind="feature",
            sign=1,
            dose=abs(n.activation) if dose == "activation" and n.activation else None,
            label=n.label,
        )
        for n in best.values()
    ]


def _version() -> str:
    from circuitcheck.validate.core import _version as v

    return v()


def _interval(t: tuple[float, float, float]) -> Interval:
    return Interval(value=t[0], low=t[1], high=t[2])


_EMPTY = Interval(value=NAN, low=NAN, high=NAN)


def _draw_controls(
    backend: SteeringBackend, mine: Sequence[LensUnit], n_random: int, rng: random.Random
) -> list[tuple[LensUnit, int]]:
    """Random units of the tested units' layer and kind: (unit, index of the tested unit it is matched to)."""
    layer = mine[0].layer
    taken = {(u.kind, u.index) for u in mine}
    out: list[tuple[LensUnit, int]] = []
    for j in range(n_random):
        t = j % len(mine)
        kind = mine[t].kind
        size = backend.n_units(layer, kind)
        if size - sum(1 for k, _ in taken if k == kind) <= 0:
            break
        idx = rng.randrange(size)
        while (kind, idx) in taken:
            idx = rng.randrange(size)
        taken.add((kind, idx))
        sign = rng.choice((1, -1)) if kind == "neuron" else 1
        out.append((LensUnit(layer=layer, index=idx, kind=kind, sign=sign), t))
    return out


def lens_predicts_steering(
    backend: SteeringBackend,
    units: Sequence[LensUnit],
    *,
    lenses: Sequence[str] | None = None,
    n_random: int = 16,
    dose_scale: float = 1.0,
    top_k: int = 10,
    label_top: int = 20,
    n_boot: int = 2000,
    seed: int = 0,
    label_judge: LabelJudge | None = None,
    progress: Callable[[str], None] | None = None,
) -> LensReport:
    """Compare each unit's lens read with its measured steering effect, against random units of the same layer.

    `units` are `LensUnit`s (use `units_from_nodes` for graph features). `lenses` defaults to every lens the
    backend offers. Each unit is steered at `dose_scale x (unit.dose or 1)`; each random unit at the dose that gives
    its push the same norm as the tested unit it is matched to. Labels are compared with the top `label_top` tokens.
    """
    say = progress or (lambda _msg: None)
    judge = label_judge or label_matches
    offered = backend.lenses()
    lens_names = list(lenses) if lenses is not None else offered
    for name in lens_names:
        if name not in offered:
            raise ValueError(f"lens {name!r} is not available; the backend offers {offered}")
    mask = backend.vocab_mask()
    seen: dict[tuple[int, int, str, int], LensUnit] = {}
    for u in units:
        seen.setdefault((u.layer, u.index, u.kind, u.sign), u)
    tested = list(seen.values())
    layers = sorted({u.layer for u in tested})
    notes: list[str] = []
    if n_random < 10:
        notes.append(f"only {n_random} random units a layer: the 'lens misses' flag needs 10")
    results: list[UnitResult] = []
    randoms: list[UnitResult] = []
    kk = max(top_k, label_top)

    for layer in layers:
        mine = [u for u in tested if u.layer == layer]
        ctrl = _draw_controls(backend, mine, n_random, random.Random(f"{seed}:{layer}"))
        all_units = [*mine, *(c for c, _ in ctrl)]
        writes = np.asarray(backend.write_vectors(all_units), dtype=np.float64)
        doses = [dose_scale * (u.dose if u.dose is not None else 1.0) for u in mine]
        norms = [abs(d) * float(np.linalg.norm(writes[i])) for i, d in enumerate(doses)]
        for j, (_, t) in enumerate(ctrl):
            wn = float(np.linalg.norm(writes[len(mine) + j]))
            doses.append(norms[t] / wn if wn > 0 else 0.0)
            norms.append(norms[t])
        push = writes * np.asarray(doses)[:, None]
        say(f"layer {layer}: steering {len(mine)} units and {len(ctrl)} random units")
        steer = np.asarray(backend.steer(layer, push), dtype=np.float64)
        keep = np.ones(steer.shape[1], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        kept_ids = np.flatnonzero(keep)
        sk = steer[:, keep]
        steer_top_ids = kept_ids[np.argsort(-sk, axis=1)[:, :kk]]
        other = np.roll(np.arange(len(all_units)), 1)
        per_lens: dict[str, dict[str, np.ndarray]] = {}
        tops: dict[str, np.ndarray] = {}
        for name in lens_names:
            lg = np.asarray(backend.lens_logits(layer, push, name), dtype=np.float64)[:, keep]
            per_lens[name] = {
                "rho": spearman_rows(lg, sk),
                "r": pearson_rows(lg, sk),
                "overlap": topk_overlap(lg, sk, top_k),
                "mis": spearman_rows(lg[other], sk) if len(all_units) > 1 else np.full(len(all_units), NAN),
            }
            tops[name] = kept_ids[np.argsort(-lg, axis=1)[:, :kk]]
        for i, u in enumerate(all_units):
            is_rand = i >= len(mine)
            scores = {
                name: UnitScore(
                    rho=float(per_lens[name]["rho"][i]),
                    r=float(per_lens[name]["r"][i]),
                    overlap=float(per_lens[name]["overlap"][i]),
                    mismatched_rho=float(per_lens[name]["mis"][i]),
                    top=backend.token_strings(tops[name][i][:top_k].tolist()),
                )
                for name in lens_names
            }
            steer_tokens = backend.token_strings(steer_top_ids[i].tolist())
            res = UnitResult(
                unit=u, random=is_rand, dose=float(doses[i]), write_norm=norms[i], scores=scores,
                steer_top=steer_tokens[:top_k],
            )  # fmt: skip
            if is_rand:
                randoms.append(res)
                continue
            if u.label:
                agrees_lens = {n: judge(u.label, backend.token_strings(tops[n][i].tolist())) for n in lens_names}
                agrees_steer = judge(u.label, steer_tokens)
                flagged = not any(agrees_lens.values()) and not agrees_steer
                res.label_check = LabelCheck(
                    label=u.label,
                    words=label_words(u.label),
                    agrees_lens=agrees_lens,
                    agrees_steering=agrees_steer,
                    flagged=flagged,
                )
                if flagged:
                    res.flags.append("label disagrees with every lens read and with the steering")
            for name in lens_names:
                rr = per_lens[name]["rho"][len(mine) :]
                low = min(float(np.quantile(rr, LOW_Q)), float(np.median(rr)) - MISS_MARGIN) if len(rr) else NAN
                if len(rr) >= 10 and scores[name].rho < low:
                    res.flags.append(
                        f"lens misses ({name}): rho {scores[name].rho:.2f} is below {1 - LOW_Q:.0%} of this layer's "
                        "random units"
                    )
            results.append(res)

    rng_np = np.random.default_rng(seed)
    layer_rows: list[LayerAgreement] = []
    summary: dict[str, LensSummary] = {}

    def med(s: Sequence[np.ndarray]) -> float:
        return float(np.median(s[0]))

    def diff(s: Sequence[np.ndarray]) -> float:
        return float(np.median(s[0]) - np.median(s[1]))

    for name in lens_names:
        samples: list[np.ndarray] = []
        weights: list[int] = []
        for layer in layers:
            tv = np.array([u.scores[name].rho for u in results if u.unit.layer == layer])
            rv = np.array([u.scores[name].rho for u in randoms if u.unit.layer == layer])
            mis = np.array([u.scores[name].mismatched_rho for u in results if u.unit.layer == layer])
            layer_rows.append(
                LayerAgreement(
                    lens=name,
                    layer=layer,
                    n=len(tv),
                    n_random=len(rv),
                    median=_interval(bootstrap(med, [tv], n_boot, rng_np)),
                    random_median=_interval(bootstrap(med, [rv], n_boot, rng_np)) if len(rv) else _EMPTY,
                    diff=_interval(bootstrap(diff, [tv, rv], n_boot, rng_np)) if len(rv) else _EMPTY,
                    mismatched_median=float(np.nanmedian(mis)) if np.isfinite(mis).any() else NAN,
                )
            )
            if len(rv):
                samples += [tv, rv]
                weights.append(len(tv))

        def overall(s: Sequence[np.ndarray], w: list[int] = weights) -> float:
            vals = [float(np.median(s[2 * i]) - np.median(s[2 * i + 1])) for i in range(len(w))]
            return float(np.average(vals, weights=w))

        d = _interval(bootstrap(overall, samples, n_boot, rng_np)) if weights else _EMPTY
        all_t = np.array([u.scores[name].rho for u in results])
        all_r = np.array([u.scores[name].rho for u in randoms])
        all_m = np.array([u.scores[name].mismatched_rho for u in results])
        m_t = float(np.median(all_t)) if len(all_t) else NAN
        m_r = float(np.median(all_r)) if len(all_r) else NAN
        m_m = float(np.nanmedian(all_m)) if np.isfinite(all_m).any() else NAN
        if not weights:
            how = "not compared (no random units)"
        elif d.low > 0:
            how = "more lens-predictable than random units of the same layers and write norm"
        elif d.high < 0:
            how = "less lens-predictable than random units of the same layers and write norm"
        else:
            how = "no more and no less lens-predictable than random units of the same layers and write norm"
        verdict = f"{name} lens: median rho {m_t:.2f} (random units {m_r:.2f}, chance {m_m:.2f}); tested units are "
        verdict += how
        summary[name] = LensSummary(
            lens=name, n=len(all_t), median=m_t, random_median=m_r, mismatched_median=m_m, diff=d, verdict=verdict
        )

    return LensReport(
        version=_version(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        backend=backend.describe(),
        lenses=lens_names,
        dose_scale=dose_scale,
        n_random=n_random,
        n_boot=n_boot,
        top_k=top_k,
        seed=seed,
        vocab_kept=int(np.asarray(mask, dtype=bool).sum()) if mask is not None else -1,
        units=results,
        random_units=randoms,
        layers=layer_rows,
        summary=summary,
        rules=RULES,
        notes=notes,
    )
