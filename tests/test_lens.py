"""The lens checks: rank agreement against random units, label flags, and graph edges against a catalog."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest
from typer.testing import CliRunner

from circuitcheck.cli import app
from circuitcheck.lens import (
    EdgeCatalog,
    LensReport,
    LensUnit,
    edges_agree,
    label_matches,
    label_words,
    lens_predicts_steering,
    neuron_poles,
    units_from_nodes,
)
from circuitcheck.lens.edges import _rewire
from circuitcheck.lens.stats import bootstrap, pearson_rows, ranks, spearman_rows, topk_overlap
from circuitcheck.lens.toy import demo, demo_edges, demo_units
from circuitcheck.report import render_edges, render_lens
from circuitcheck.validate import FeatureNode

runner = CliRunner()


# ------------------------------------------------------------------ stats


def test_ranks_average_ties() -> None:
    assert ranks(np.array([[3.0, 1.0, 2.0]])).tolist() == [[2.0, 0.0, 1.0]]
    assert ranks(np.array([[1.0, 1.0, 5.0]])).tolist() == [[0.5, 0.5, 2.0]]


def test_spearman_is_rank_pearson() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 50))
    b = a**3  # monotone in a: rho is 1, Pearson below it
    rho = spearman_rows(a, b)
    assert rho.tolist() == pytest.approx([1.0] * 3)
    assert np.all(pearson_rows(a, b) < 0.99)
    assert spearman_rows(a, -a).tolist() == pytest.approx([-1.0] * 3)
    assert pearson_rows(np.ones((1, 5)), a[:1, :5]).tolist() == [0.0]


def test_topk_overlap_and_bootstrap() -> None:
    assert topk_overlap(np.array([[5, 4, 3, 0, 0]]), np.array([[5, 0, 4, 3, 0]]), 2).tolist() == [0.5]
    x = np.arange(100, dtype=float)
    v, lo, hi = bootstrap(lambda s: float(np.median(s[0])), [x], 500, np.random.default_rng(0))
    assert lo < v < hi and v == 49.5


# ------------------------------------------------------------------ labels


def test_label_words_and_matching() -> None:
    assert label_words("Austin, the capital of Texas") == ["austin", "capital", "texas"]
    assert ")" not in label_words("code closers: ')' before newlines")  # one ASCII character: too short
    assert label_matches("pet animals", [" dogs", " animals"])
    assert label_matches("dogs and cats", [" dog"])  # prefix either way, 3+ characters
    assert not label_matches("Texas cities", [" Paris", " France"])
    assert not label_matches("the words", [" the"])  # stop words never match
    assert label_matches("植物 and gardens", ["植物"])  # non-ASCII words of any length
    assert label_matches(
        "semicolon endings",
        [
            ";",
        ],
    )  # punctuation names name the characters
    assert label_matches("long runs of spaces", ["      "])
    assert label_matches("trailing punctuation", [")?"])
    assert not label_matches("semicolon endings", [" the", ":"])


def test_units_from_nodes_keeps_largest_activation() -> None:
    nodes = [
        FeatureNode(layer=3, pos=4, feature=7, activation=2.0, label="x"),
        FeatureNode(layer=3, pos=6, feature=7, activation=5.0, label="x"),
        FeatureNode(layer=4, pos=6, feature=1, activation=0.0),
    ]
    units = units_from_nodes(nodes)
    assert [(u.layer, u.index, u.dose) for u in units] == [(3, 7, 5.0), (4, 1, None)]
    assert units[0].label == "x"
    assert [u.dose for u in units_from_nodes(nodes, dose="unit")] == [None, None]


# ------------------------------------------------------------------ lens_predicts_steering on the toy


@pytest.fixture(scope="module")
def toy_report() -> LensReport:
    return lens_predicts_steering(demo(), demo_units(), n_random=16, n_boot=300, label_top=3)


def _layer(rep: LensReport, lens: str, layer: int):
    return next(r for r in rep.layers if r.lens == lens and r.layer == layer)


def test_toy_lenses_behave_as_planted(toy_report: LensReport) -> None:
    """The logit lens reads layer 1 badly and the last layer well; the R-lens (the average map) reads both well."""
    assert _layer(toy_report, "logit", 1).median.value < 0.3
    assert _layer(toy_report, "logit", 5).median.value > 0.9
    assert _layer(toy_report, "R", 1).median.value > 0.9
    for name in ("logit", "R"):
        s = toy_report.summary[name]
        assert s.mismatched_median < 0.3  # another unit's read predicts nothing
        assert "no more and no less" in s.verdict  # the toy's graph units are ordinary units
        assert s.diff.low <= 0 <= s.diff.high


def test_toy_controls_are_matched(toy_report: LensReport) -> None:
    assert len(toy_report.units) == 14 and len(toy_report.random_units) == 3 * 16
    norms: dict[int, list[float]] = {}
    for u in toy_report.units:
        norms.setdefault(u.unit.layer, []).append(u.write_norm)
    for r in toy_report.random_units:
        assert any(math.isclose(r.write_norm, n) for n in norms[r.unit.layer])
        assert r.unit.index not in {u.unit.index for u in toy_report.units if u.unit.layer == r.unit.layer}


def test_toy_label_flags(toy_report: LensReport) -> None:
    by = {u.unit.name(): u for u in toy_report.units}
    austin, paris = by["L5/0"], by["L5/1"]
    assert austin.steer_top[0] == " Austin" and austin.label_check and not austin.label_check.flagged
    assert paris.steer_top[0] == " Paris" and paris.label_check and paris.label_check.flagged
    assert "label disagrees with every lens read and with the steering" in paris.flags
    assert all(not u.label_check for u in toy_report.units if u.unit.label is None)


def test_toy_report_round_trips_and_renders(toy_report: LensReport) -> None:
    again = LensReport.model_validate_json(toy_report.model_dump_json())
    assert again.summary["R"].median == toy_report.summary["R"].median
    page = render_lens(toy_report)
    assert "<!doctype html>" in page and "Texas cities" in page and "By layer" in page and "chance" in page


def test_same_seed_same_controls() -> None:
    a = lens_predicts_steering(demo(), demo_units(), n_random=8, n_boot=50, seed=3)
    b = lens_predicts_steering(demo(), demo_units(), n_random=8, n_boot=50, seed=3)
    assert [r.unit for r in a.random_units] == [r.unit for r in b.random_units]
    assert a.notes and "needs 10" in a.notes[0]  # too few random units for the 'lens misses' flag


def test_unknown_lens_is_an_error() -> None:
    with pytest.raises(ValueError, match="not available"):
        lens_predicts_steering(demo(), demo_units(), lenses=["J"])


class _Clean:
    """Steering equals the lens read for units 0-3 and is mostly noise for every other unit."""

    def __init__(self) -> None:
        rng = np.random.default_rng(1)
        self.read = rng.normal(size=(100, 40))
        self.noise = rng.normal(size=(100, 40))
        self._last: list[int] = []

    def write_vectors(self, units: Sequence[LensUnit]) -> np.ndarray:
        self._last = [u.index for u in units]
        return np.ones((len(units), 4))

    def n_units(self, layer: int, kind: str) -> int:
        return 100

    def lenses(self) -> list[str]:
        return ["logit"]

    def lens_logits(self, layer: int, vectors: np.ndarray, lens: str) -> np.ndarray:
        return self.read[self._last]

    def steer(self, layer: int, vectors: np.ndarray) -> np.ndarray:
        return np.stack([self.read[i] + (0.0 if i < 4 else 3.0) * self.noise[i] for i in self._last])

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [f"t{i}" for i in ids]

    def vocab_mask(self) -> np.ndarray | None:
        return None

    def describe(self) -> dict[str, str]:
        return {"model": "clean"}


def test_verdict_says_more_predictable_when_tested_units_are() -> None:
    rep = lens_predicts_steering(_Clean(), [LensUnit(layer=2, index=i) for i in range(4)], n_random=20, n_boot=300)
    s = rep.summary["logit"]
    assert s.median == pytest.approx(1.0) and s.random_median < 0.6
    assert s.diff.low > 0 and "more lens-predictable" in s.verdict


# ------------------------------------------------------------------ edges_agree


def test_neuron_poles() -> None:
    assert neuron_poles("3:17") == [(3, 17, 1), (3, 17, -1)]
    assert neuron_poles("3:17-") == [(3, 17, -1)]


def _rows() -> list[dict]:
    """Two poles in neuron-lens's `edges` format, lists padded with layer -1 and NaN."""
    nan = float("nan")
    return [
        {"layer": 1, "neuron": 5, "pole": 1,
         "in_layer": [0, -1, -1], "in_neuron": [9, -1, -1], "in_pole": [-1, 0, 0], "in_push": [0.7, nan, nan],
         "out_layer": [2, 3, -1], "out_neuron": [4, 8, -1], "out_pole": [1, -1, 0], "out_push": [-0.4, 0.9, nan]},
        {"layer": 2, "neuron": 4, "pole": 1,
         "in_layer": [1, -1, -1], "in_neuron": [5, -1, -1], "in_pole": [1, 0, 0], "in_push": [-0.5, nan, nan],
         "out_layer": [-1, -1, -1], "out_neuron": [-1, -1, -1], "out_pole": [0, 0, 0], "out_push": [nan, nan, nan]},
    ]  # fmt: skip


def test_catalog_from_rows_and_json(tmp_path) -> None:
    cat = EdgeCatalog.from_rows(_rows(), name="fixture")
    assert cat.get((0, 9, -1), (1, 5, 1)) == pytest.approx(0.7)
    assert cat.get((1, 5, 1), (3, 8, -1)) == pytest.approx(0.9)
    assert cat.get((1, 5, 1), (2, 4, 1)) == pytest.approx(-0.5)  # listed by both poles; the larger |push| is kept
    assert len(cat) == 3 and cat.layers == {1, 2}
    cat.save(tmp_path / "c.json")
    again = EdgeCatalog.load(tmp_path / "c.json")
    assert again.pairs == cat.pairs and again.layers == cat.layers
    sources = EdgeCatalog.from_rows(
        [{"layer": 2, "neuron": 4, "pole": 1, "driven_layer": [1, -1], "driven_neuron": [5, -1],
          "driven_pole": [1, 0], "driven_share": [0.3, float("nan")]}],
        table="sources",
    )  # fmt: skip
    assert sources.pairs == {((1, 5, 1), (2, 4, 1)): pytest.approx(0.3)}
    with pytest.raises(ValueError, match="table"):
        EdgeCatalog.from_rows([], table="wiring")


def test_skips_are_counted() -> None:
    cat = EdgeCatalog.from_rows(_rows())
    graph = {
        ("1:5+", "2:4+"): -1.0,  # present, same sign as the catalog's push
        ("1:5", "3:8"): 1.0,  # layer 3 not loaded
        ("2:4", "1:5"): 1.0,  # backward
        ("1:5", "2:7"): 1.0,  # compared, absent
    }
    rep = edges_agree(graph, cat, n_null=20)
    assert rep.skipped == {"unmapped": 0, "layer not covered": 1, "not forward": 1}
    assert rep.n_compared == 2 and rep.n_present == 1 and rep.share == 0.5
    assert rep.present[0].via == ((1, 5, 1), (2, 4, 1)) and rep.sign_agreement == 1.0
    rep2 = edges_agree(graph, cat, unit_map={"1:5+": [(1, 5, 1)]}, n_null=5)
    assert rep2.skipped["unmapped"] == 4 and rep2.verdict == "no edges could be compared"


def test_demo_graph_follows_its_catalog() -> None:
    graph, cat = demo_edges()
    rep = edges_agree(graph, cat, n_null=200)
    assert rep.share > 0.4 and rep.null.high < 0.15
    assert rep.p_value < 0.01 and rep.enrichment and rep.enrichment > 4
    assert rep.sign_agreement == 1.0 and rep.rewired > 0.5
    assert "follows" in rep.verdict
    assert sum(g.n for g in rep.by_gap) == rep.n_compared
    page = render_edges(rep)
    assert "<!doctype html>" in page and "rewirings" in page


def test_another_graph_does_not_follow_the_catalog() -> None:
    _, cat = demo_edges()
    graph, _ = demo_edges(seed=1)  # another seed's graph: its catalog edges are not this catalog's
    rep = edges_agree(graph, cat, n_null=200)
    assert rep.p_value > 0.05 and "no more than" in rep.verdict


def test_rewiring_keeps_degrees_and_layer_order() -> None:
    graph, _ = demo_edges()
    es = list(graph)
    work = list(es)
    assert _rewire(work, 5 * len(work), random.Random(0)) > 0
    assert Counter(a for a, _ in work) == Counter(a for a, _ in es)
    assert Counter(b for _, b in work) == Counter(b for _, b in es)
    assert len(set(work)) == len(work)
    assert all(int(a.split(":")[0]) < int(b.split(":")[0]) for a, b in work)


# ------------------------------------------------------------------ CLI


def test_cli_lens_and_edges_demo(tmp_path) -> None:
    out = runner.invoke(app, ["lens", "--demo", "--n-boot", "100", "--json", str(tmp_path / "l.json")])
    assert out.exit_code == 0, out.output
    assert "flagged L5/1" in out.output
    assert json.loads((tmp_path / "l.json").read_text())["summary"]["R"]["median"] > 0.9
    out = runner.invoke(app, ["edges", "--demo", "--html", str(tmp_path / "e.html")])
    assert out.exit_code == 0, out.output
    assert "follows" in out.output and (tmp_path / "e.html").exists()


@pytest.mark.slow
def test_neuron_lens_catalog_downloads() -> None:
    """Network: neuron-lens's Qwen3-0.6B edges and sources for two layers load and link them."""
    from circuitcheck.lens.neuron_lens import load_catalog

    cat = load_catalog("Qwen/Qwen3-0.6B", [5, 6])
    assert cat.layers == {5, 6} and len(cat) > 100_000
    assert any(a[0] == 5 and b[0] == 6 for a, b in cat.pairs)
    src = load_catalog("Qwen/Qwen3-0.6B", [5, 6], table="sources")
    assert all(b[0] > a[0] for a, b in src.pairs)
