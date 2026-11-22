"""The saved results of the live GPU runs (examples/modal/live_checks.py) parse, and their verdicts follow the rules.

These re-check the recorded runs offline: every stored verdict must be what `decide()` gives on the stored checks,
and the headline findings of the circuits episodes must hold.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from circuitcheck.validate import Report, Verdict, decide


def _load(data_dir, job: str) -> dict:
    return json.loads((data_dir / "live" / f"{job}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("job", ["rl_hack", "lock"])
def test_stored_verdicts_follow_the_rules(data_dir, job) -> None:
    for raw in _load(data_dir, job)["validate"]:
        rep = Report.model_validate(raw)
        assert rep.n_controls >= 19
        for g in rep.groups:
            verdict, _ = decide(g.ablation, g.steering, g.patch, rep.thresholds)
            assert verdict == g.verdict, f"{job} {g.name}"


def test_rl_hack_group_is_a_readout(data_dir) -> None:
    """Episode 2: the graph's ten RL features, switched off together, do not carry the hack's next token.

    One feature on its own (26:88470) meets the switch rule at one of the three slots, at the smallest p-value 20
    controls allow; with 11 groups x 3 slots that is about what chance gives (33 x 0.05 = 1.6), and it is the only one.
    """
    res = _load(data_dir, "rl_hack")
    assert len(res["validate"]) == 3
    singles = []
    for raw in res["validate"]:
        rep = Report.model_validate(raw)
        group = next(g for g in rep.groups if g.name == "graph's top 10")
        assert group.verdict == Verdict.READOUT
        singles += [g.name for g in rep.groups if g is not group and g.verdict == Verdict.SWITCH]
    assert singles == ["26:88470"]


def test_rl_hack_reconstruction_matches_study(data_dir) -> None:
    res = _load(data_dir, "rl_hack")
    before, after = np.mean(res["fvu"]["before"]), np.mean(res["fvu"]["after"])
    assert before == pytest.approx(0.500, abs=0.005) and after == pytest.approx(0.499, abs=0.005)
    for s in res["summaries"].values():
        assert s["top20_overlap_with_study"] >= 18
        assert s["replacement"] == pytest.approx(s["replacement_study"], abs=0.02)


def test_lock_claimed_group_is_not_a_switch(data_dir) -> None:
    """Episode 1: switching off the ten claimed lock features does not unlock the model."""
    res = _load(data_dir, "lock")
    assert res["validate"]
    for raw in res["validate"]:
        rep = Report.model_validate(raw)
        group = next(g for g in rep.groups if g.name == "claimed 10")
        assert group.verdict not in (Verdict.SWITCH, Verdict.PARTIAL)


def test_lens_qwen3_reports_parse_and_follow_neuron_lens(data_dir) -> None:
    """Qwen3-0.6B (examples/modal/lens_checks.py): the R-lens predicts steering far better than the logit lens before
    the last layers, as neuron-lens's findings §3 report for Qwen3-8B, and both beat the mismatched-pair chance
    level."""
    from circuitcheck.lens import LensReport

    res = _load(data_dir, "lens_qwen3-0.6b")
    for key in ("lens_features", "lens_neurons"):
        rep = LensReport.model_validate(res[key])
        assert rep.lenses == ["logit", "R"] and rep.n_random == 16
        assert rep.summary["R"].median > rep.summary["logit"].median + 0.2
        for s in rep.summary.values():
            assert abs(s.mismatched_median) < 0.1
            assert s.median > s.mismatched_median + 0.3
    neurons = LensReport.model_validate(res["lens_neurons"])
    early = [r for r in neurons.layers if r.layer <= 13]
    logit = np.median([r.median.value for r in early if r.lens == "logit"])
    r_lens = np.median([r.median.value for r in early if r.lens == "R"])
    assert r_lens > logit + 0.25
    assert all(u.label_check for u in neurons.units)  # every sampled pole has a neuron-lens label


def test_lens_qwen3_graph_edges_follow_the_catalog(data_dir) -> None:
    """Pooled over the three graphs, the catalog lists more graph edges than degree-matched rewirings do."""
    from circuitcheck.lens import EdgesReport

    res = _load(data_dir, "lens_qwen3-0.6b")
    for table in ("edges", "sources"):
        rep = EdgesReport.model_validate(res["edges"][f"pooled/k1/{table}"])
        assert rep.n_compared == rep.n_edges > 10_000
        assert rep.share > rep.null.high and rep.p_value <= 0.01
    assert res["map_cos"]["k1"]["median"] > 5 * res["map_cos"]["k1"]["one_over_sqrt_d"]
