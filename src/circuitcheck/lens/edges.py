"""Do an attribution graph's edges appear in a whole-model edge catalog? Against degree-matched rewirings.

An attribution graph's edges say which unit fed which *on one prompt*. neuron-lens (github.com/stdlogits/neuron-lens,
MIT; docs/findings.md §12 "What feeds a unit: edges and sources", `scripts/checks/edges_in_context.py`) publishes two
whole-model tables per MLP neuron pole of Qwen3 (0.6B-14B) and Gemma 4 12B:

- `edges`: each pole's 64 strongest direct weight edges from earlier poles and to later ones (the cosine of one
  pole's write with the other's read direction, weighted by its typical firing): wiring, from the weights alone;
- `sources`: the 64 upstream poles that carry most of a pole's activation at its top firings: traffic, in context.

`edges_agree` maps both ends of every graph edge to catalog poles, counts the graph edges the catalog also lists, and
compares that share with the share in rewired copies of the graph that keep every unit's in- and out-degree and the
layer order (Maslov-Sneppen swaps). A graph whose edges follow the model's wiring beats its rewirings.

Graph units are `"layer:index"` strings, as in `circuitcheck.diff` summaries. For a graph over MLP neurons the default
map takes both poles of each neuron (`"L:n"`) or one (`"L:n+"`, `"L:n-"`). For transcoder features, pass a map from
`circuitcheck.tracer.map_features_to_neurons` (each feature's write matched to its nearest neuron writes).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from circuitcheck.lens.types import EdgeMatch, EdgesReport, GapRow, Interval, Pole

GAPS: list[tuple[str, int, int]] = [("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10**6)]
NAN = float("nan")

RULES = [
    "compared: a graph edge whose two units map to catalog poles of covered layers, the source's layer earlier",
    "present: some (source pole, target pole) pair of the two maps is in the catalog",
    "null: degree-matched rewirings of the compared edges (Maslov-Sneppen swaps that keep every unit's in- and "
    "out-degree and send edges only to later layers), and the share present in each",
    "p-value: (1 + rewirings with a share at least the observed one) / (rewirings + 1)",
    "verdict: the graph follows the catalog when p <= 0.05; the catalog lists fewer of its edges than of rewirings' "
    "when the share is below the null's 2.5% point",
]


class EdgeCatalog:
    """Directed pole-to-pole edges with a weight (`push` for edges, activation `share` for sources)."""

    def __init__(
        self, pairs: Mapping[tuple[Pole, Pole], float], layers: Iterable[int], name: str = "", table: str = "edges"
    ) -> None:
        if table not in ("edges", "sources"):
            raise ValueError(f"table must be 'edges' or 'sources', not {table!r}")
        self.pairs: dict[tuple[Pole, Pole], float] = dict(pairs)
        self.layers = set(layers)
        self.name = name
        self.table = table

    def __len__(self) -> int:
        return len(self.pairs)

    def covers(self, layer: int) -> bool:
        return layer in self.layers

    def get(self, a: Pole, b: Pole) -> float | None:
        return self.pairs.get((a, b))

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]], table: str = "edges", name: str = "") -> EdgeCatalog:
        """Build from rows of neuron-lens's `<model>-neurons-edges` or `-sources` tables (one dict per pole).

        edges: `in_layer/in_neuron/in_pole/in_push` (earlier poles into this one) and `out_*` (this one into later
        poles). sources: `driven_layer/driven_neuron/driven_pole/driven_share` (earlier poles that drive this one
        at its top firings). Lists are padded with layer -1.
        """
        if table not in ("edges", "sources"):
            raise ValueError(f"table must be 'edges' or 'sources', not {table!r}")
        sides = [("in", "push", True), ("out", "push", False)] if table == "edges" else [("driven", "share", True)]
        pairs: dict[tuple[Pole, Pole], float] = {}
        layers: set[int] = set()
        for row in rows:
            me: Pole = (int(row["layer"]), int(row["neuron"]), int(row["pole"]))
            layers.add(me[0])
            for prefix, val, incoming in sides:
                cols = zip(
                    row[f"{prefix}_layer"], row[f"{prefix}_neuron"], row[f"{prefix}_pole"], row[f"{prefix}_{val}"],
                    strict=True,
                )  # fmt: skip
                for lay, neu, pol, w in cols:
                    if int(lay) < 0 or w != w:
                        continue
                    other: Pole = (int(lay), int(neu), int(pol))
                    key = (other, me) if incoming else (me, other)
                    if key not in pairs or abs(float(w)) > abs(pairs[key]):
                        pairs[key] = float(w)
        return cls(pairs, layers, name=name, table=table)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "table": self.table,
            "layers": sorted(self.layers),
            "pairs": [[*a, *b, w] for (a, b), w in self.pairs.items()],
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> EdgeCatalog:
        pairs = {
            ((int(p[0]), int(p[1]), int(p[2])), (int(p[3]), int(p[4]), int(p[5]))): float(p[6]) for p in d["pairs"]
        }
        return cls(pairs, d["layers"], name=d.get("name", ""), table=d.get("table", "edges"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EdgeCatalog:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def neuron_poles(unit: str) -> list[Pole]:
    """The default map for graphs over MLP neurons: `"L:n"` -> both poles, `"L:n+"` / `"L:n-"` -> one."""
    layer, rest = unit.split(":")
    if rest.endswith(("+", "-")):
        return [(int(layer), int(rest[:-1]), 1 if rest.endswith("+") else -1)]
    return [(int(layer), int(rest), 1), (int(layer), int(rest), -1)]


def _layer(unit: str) -> int:
    return int(unit.split(":")[0])


def _gap(g: int) -> str:
    return next(name for name, lo, hi in GAPS if lo <= g <= hi)


def _rewire(edges: list[tuple[str, str]], n_swaps: int, rng: random.Random) -> int:
    """Maslov-Sneppen swaps in place: (a->b, c->d) becomes (a->d, c->b) when both stay forward and new. Returns the
    number of swaps made."""
    m = len(edges)
    if m < 2:
        return 0
    present = set(edges)
    done = 0
    for _ in range(n_swaps):
        i, j = rng.randrange(m), rng.randrange(m)
        (a, b), (c, d) = edges[i], edges[j]
        if i == j or a == c or b == d:
            continue
        if _layer(a) >= _layer(d) or _layer(c) >= _layer(b):
            continue
        if (a, d) in present or (c, b) in present:
            continue
        present -= {(a, b), (c, d)}
        present |= {(a, d), (c, b)}
        edges[i], edges[j] = (a, d), (c, b)
        done += 1
    return done


def edges_agree(
    graph_edges: Mapping[tuple[str, str], float] | Any,
    catalog: EdgeCatalog,
    *,
    unit_map: Mapping[str, Sequence[Pole]] | Callable[[str], Sequence[Pole]] | None = None,
    top: int | None = None,
    n_null: int = 200,
    seed: int = 0,
    notes: Sequence[str] = (),
) -> EdgesReport:
    """Share of the graph's edges the catalog also lists, against degree-matched rewirings of the graph.

    `graph_edges` maps (source unit, target unit) to the edge weight, or is a `GraphSummary` (its `edges` are used).
    `top` keeps the strongest edges by |weight|. `unit_map` maps a unit to its catalog poles (default
    `neuron_poles`).
    """
    edges_in: dict[tuple[str, str], float] = dict(getattr(graph_edges, "edges", graph_edges))
    if top is not None:
        edges_in = dict(sorted(edges_in.items(), key=lambda kv: -abs(kv[1]))[:top])
    if unit_map is None:
        poles_of: Callable[[str], Sequence[Pole]] = neuron_poles
    elif callable(unit_map):
        poles_of = unit_map
    else:
        mp = unit_map

        def poles_of(u: str) -> Sequence[Pole]:
            return mp.get(u, ())

    cache: dict[str, list[Pole]] = {}

    def poles(u: str) -> list[Pole]:
        if u not in cache:
            cache[u] = list(poles_of(u))
        return cache[u]

    memo: dict[tuple[str, str], tuple[Pole, Pole, float] | None] = {}

    def match(a: str, b: str) -> tuple[Pole, Pole, float] | None:
        if (a, b) not in memo:
            best: tuple[Pole, Pole, float] | None = None
            for pa in poles(a):
                for pb in poles(b):
                    w = catalog.get(pa, pb)
                    if w is not None and (best is None or abs(w) > abs(best[2])):
                        best = (pa, pb, w)
            memo[(a, b)] = best
        return memo[(a, b)]

    skipped = {"unmapped": 0, "layer not covered": 0, "not forward": 0}
    compared: list[tuple[str, str]] = []
    for a, b in edges_in:
        pa, pb = poles(a), poles(b)
        if not pa or not pb:
            skipped["unmapped"] += 1
        elif not all(catalog.covers(p[0]) for p in [*pa, *pb]):
            skipped["layer not covered"] += 1
        elif _layer(a) >= _layer(b):
            skipped["not forward"] += 1
        else:
            compared.append((a, b))

    def share_of(es: Sequence[tuple[str, str]]) -> tuple[float, dict[str, list[bool]]]:
        hits = [match(a, b) is not None for a, b in es]
        gaps: dict[str, list[bool]] = {}
        for (a, b), h in zip(es, hits, strict=True):
            gaps.setdefault(_gap(_layer(b) - _layer(a)), []).append(h)
        return (float(np.mean(hits)) if hits else 0.0), gaps

    share, gaps = share_of(compared)
    rng = random.Random(seed)
    work = list(compared)
    _rewire(work, 5 * len(work), rng)
    null: list[float] = []
    null_gaps: dict[str, list[float]] = {}
    changed: list[float] = []
    original = set(compared)
    for _ in range(n_null if len(compared) >= 2 else 0):
        _rewire(work, len(work), rng)
        s, g = share_of(work)
        null.append(s)
        for k, v in g.items():
            null_gaps.setdefault(k, []).append(float(np.mean(v)))
        changed.append(sum(e not in original for e in work) / len(work))
    arr = np.asarray(null)
    if len(arr):
        nmean = float(arr.mean())
        lo, hi = (float(x) for x in np.quantile(arr, [0.025, 0.975]))
        p = (1 + int((arr >= share).sum())) / (len(arr) + 1)
    else:
        nmean, lo, hi, p = NAN, NAN, NAN, 1.0

    present: list[EdgeMatch] = []
    for a, b in compared:
        m = match(a, b)
        if m is not None:
            w = float(edges_in[(a, b)])
            present.append(
                EdgeMatch(source=a, target=b, weight=w, catalog_weight=m[2], via=(m[0], m[1]), same_sign=w * m[2] > 0)
            )
    present.sort(key=lambda x: -abs(x.weight))
    sign = float(np.mean([x.same_sign for x in present])) if present else None
    by_gap = [
        GapRow(
            gap=name,
            n=len(gaps[name]),
            share=float(np.mean(gaps[name])),
            null_share=float(np.mean(null_gaps[name])) if null_gaps.get(name) else NAN,
        )
        for name, _, _ in GAPS
        if gaps.get(name)
    ]
    rewired = float(np.mean(changed)) if changed else 0.0
    all_notes = list(notes)
    if compared and rewired < 0.2:
        all_notes.append(
            f"rewiring changed only {rewired:.0%} of the edges: the graph's degrees leave little freedom, so the null "
            "is close to the graph itself"
        )
    has_null = nmean == nmean
    if not compared:
        verdict = "no edges could be compared"
    elif p <= 0.05 and has_null:
        ratio = f"x{share / nmean:.1f}, " if nmean > 0 else ""
        verdict = (
            f"the catalog lists {share:.1%} of the graph's edges against {nmean:.1%} of rewired graphs' "
            f"({ratio}p = {p:.3f}): the graph follows the model's {catalog.table}"
        )
    elif has_null and share < lo:
        verdict = f"the catalog lists fewer of the graph's edges ({share:.1%}) than of rewired graphs' ({nmean:.1%})"
    elif has_null:
        verdict = (
            f"the catalog lists {share:.1%} of the graph's edges, no more than rewired graphs' {nmean:.1%} "
            f"(p = {p:.3f})"
        )
    else:
        verdict = f"the catalog lists {share:.1%} of the graph's edges; too few edges to rewire"
    from circuitcheck.lens.steering import _version

    return EdgesReport(
        version=_version(),
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        catalog={
            "name": catalog.name,
            "table": catalog.table,
            "pairs": str(len(catalog)),
            "layers": f"{min(catalog.layers)}-{max(catalog.layers)}" if catalog.layers else "",
        },
        n_edges=len(edges_in),
        n_compared=len(compared),
        n_present=len(present),
        share=share,
        null=Interval(value=nmean, low=lo, high=hi),
        p_value=p,
        enrichment=share / nmean if has_null and nmean > 0 else None,
        sign_agreement=sign,
        rewired=rewired,
        n_null=len(null),
        seed=seed,
        by_gap=by_gap,
        present=present[:50],
        skipped=skipped,
        verdict=verdict,
        rules=RULES,
        notes=all_notes,
    )
