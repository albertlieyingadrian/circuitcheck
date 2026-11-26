"""circuitcheck's tools on raw MLP neurons, on a tiny transformer with planted neurons (CPU, a few seconds).

The toy (`circuitcheck.neurons.toy`) has a switch neuron L0/N0 and a readout neuron L1/N0 with known roles, and a
fine-tuned copy in which L1/N5 learned to fire on `D`. Tests that need torch, transformers and neuronkit skip
cleanly without them.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from circuitcheck.neurons import parse_neuron, parse_pole


def test_parse_specs() -> None:
    assert parse_neuron("12:345@9") == (12, 345, 9)
    assert parse_neuron("L12/N345@-1") == (12, 345, -1)
    assert parse_neuron((3, 4, 5)) == (3, 4, 5)
    with pytest.raises(ValueError):
        parse_neuron("12:345")
    assert parse_pole("12:n345+") == (12, 345, 1)
    assert parse_pole("12:345-") == (12, 345, -1)
    assert parse_pole("L2/N7") == (2, 7, 1)


def test_import_stays_light() -> None:
    """circuitcheck (and its neuron module) import without torch or neuronkit: both load only when used."""
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['neuronkit'] = None; sys.modules['transformers'] = None\n"
        "import circuitcheck.neurons, circuitcheck.report, circuitcheck.cli"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


nk = pytest.importorskip("neuronkit")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from circuitcheck.neurons import (  # noqa: E402
    NEURON_CAVEAT,
    NeuronBackend,
    diff_neurons,
    label_neurons,
    neuron_summaries,
    tail_thresholds,
    toy,
)
from circuitcheck.validate import GroupSpec, Verdict, logit_diff, validate  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    return toy.demo()


def _metric(b: NeuronBackend):
    return logit_diff(b.token_id(toy.TARGET), b.token_id(toy.CONTRAST), toy.TARGET, toy.CONTRAST)


@pytest.fixture(scope="module")
def report(planted):
    b, _, _ = planted
    nodes = b.nodes(toy.PROMPT, toy.demo_specs())
    return validate(b, toy.PROMPT, nodes, metric=_metric(b), counterfactual=toy.COUNTERFACTUAL, n_controls=20)


# ---------------------------------------------------------------- validate


def test_toy_verdicts_match_the_planted_roles(report) -> None:
    last = len(toy.PROMPT) - 1
    by = {g.name: g.verdict for g in report.groups}
    assert by[f"L{toy.SWITCH[0]}/{toy.SWITCH[1]}@{last}"] == Verdict.SWITCH
    assert by[f"L{toy.READOUT[0]}/{toy.READOUT[1]}@{last}"] == Verdict.READOUT
    assert by["top-2"] == Verdict.SWITCH
    assert report.backend["basis"].startswith("MLP neurons")


def test_readout_is_active_but_idle(report) -> None:
    """The readout fires as hard as the switch, but switching it off does not move the output."""
    sw, ro = report.groups[1], report.groups[2]
    assert ro.nodes[0].activation > 0.5 * sw.nodes[0].activation
    assert abs(ro.ablation.effect) < 0.05 * abs(sw.ablation.effect)
    assert sw.patch is not None and sw.patch.gap_closed > 0.3


def test_nodes_carry_real_activations(planted) -> None:
    b, m, _ = planted
    node = b.nodes(toy.PROMPT, ["0:0@-1"])[0]
    with nk.capture(m, neurons=[nk.Neuron(0, 0)]) as st, torch.no_grad():
        m(input_ids=torch.tensor([b.ids(toy.PROMPT)]), use_cache=False)
    assert node.pos == len(toy.PROMPT) - 1
    assert node.activation == pytest.approx(float(st[0][0][0, -1, 0]), rel=1e-5)


def test_logits_edit_matches_neuronkit_ablate(planted) -> None:
    """Setting a neuron to 0 at every position through the backend equals neuronkit's own ablation."""
    b, m, _ = planted
    n = len(b.ids(toy.PROMPT))
    ours = b.logits(toy.PROMPT, [(0, p, 0, 0.0) for p in range(n)])
    with nk.ablate(m, [nk.Neuron(0, 0)]), torch.no_grad():
        ref = m(input_ids=torch.tensor([b.ids(toy.PROMPT)]), use_cache=False).logits[0].numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-5)
    assert not np.allclose(ours, b.logits(toy.PROMPT), atol=1e-3)


def test_edits_stay_at_their_position(planted) -> None:
    """An edit at the last position leaves every earlier position's logits unchanged (attention is causal)."""
    b, _, _ = planted
    clean = b.logits(toy.PROMPT)
    edited = b.logits(toy.PROMPT, [(0, len(toy.PROMPT) - 1, 0, 0.0)])
    np.testing.assert_allclose(edited[:-1], clean[:-1], atol=1e-6)


def test_controls_come_from_the_tested_layers(planted) -> None:
    _, m, _ = planted
    b2 = NeuronBackend(m, toy.CharTokenizer())
    b2.nodes(toy.PROMPT, ["1:0@-1"])
    layers = {a[0] for a in b2.active_features(toy.PROMPT)}
    assert layers == {1}
    w = nk.sites(m)[1].down_cols()
    assert b2.decoder_norm(1, 3) == pytest.approx(float(w[3].norm()), rel=1e-6)


def test_subcircuit_drivers_of_the_readout_are_a_switch(planted) -> None:
    """The readout's strongest driver is the switch: validating the drivers finds the switch one layer down."""
    b, _, _ = planted
    group, sc = b.subcircuit_group(toy.PROMPT, f"{toy.READOUT[0]}:{toy.READOUT[1]}@-1", k=1)
    assert [(n.layer, n.feature) for n in group.nodes] == [toy.SWITCH]
    assert sc.completeness == pytest.approx(1.0, abs=1e-3)
    rep = validate(
        b,
        toy.PROMPT,
        group.nodes,
        metric=_metric(b),
        counterfactual=toy.COUNTERFACTUAL,
        groups=[group],
        n_controls=20,
    )
    assert rep.groups[0].verdict == Verdict.SWITCH


def test_validate_takes_neuron_groups(planted) -> None:
    b, _, _ = planted
    nodes = b.nodes(toy.PROMPT, toy.demo_specs())
    rep = validate(
        b,
        toy.PROMPT,
        nodes,
        metric=_metric(b),
        groups=[GroupSpec(name="both", nodes=nodes)],
        n_controls=20,
    )
    assert [g.name for g in rep.groups] == ["both"]
    assert rep.groups[0].verdict == Verdict.SWITCH  # no counterfactual: necessary and specific is enough


# ---------------------------------------------------------------- labels


@pytest.fixture(scope="module")
def labels(planted):
    _, m, tok = planted
    return label_neurons(m, tok, toy.CORPUS, ["0:n0+", "1:n0+"], llm=toy.judge(), n_explain=4, n_write=2)


def test_label_describes_and_scores(labels) -> None:
    sw = labels[0]
    assert sw.error is None and "letter D" in sw.label
    assert sw.recovered is not None and sw.recovered > 0.9
    assert sw.predict_accuracy is not None and sw.predict_accuracy > 0.7
    assert sw.detection is not None and sw.detection.balanced_accuracy == sw.predict_accuracy
    assert sw.score == sw.recovered
    assert sw.n_windows > 0 and sw.write_best is not None


def test_readout_label_has_no_damage_to_recover(labels) -> None:
    """Ablating the readout does nothing, so there is no damage for a description to recover: the trust score is
    None and the ablation KL says why."""
    ro = labels[1]
    assert ro.error is None and ro.recovered is None
    assert ro.ablation_kl is not None and ro.ablation_kl < 1e-6


def test_wrong_description_earns_nothing(planted) -> None:
    _, m, tok = planted
    wrong = label_neurons(
        m, tok, toy.CORPUS, ["0:n0+"], llm=toy.judge(), descriptions={"0:n0+": "the letter q"}, n_explain=4
    )[0]
    assert wrong.label == "the letter q"
    assert wrong.recovered is not None and wrong.recovered < 0.1
    assert wrong.predict_accuracy == pytest.approx(0.5)


def test_label_error_is_kept(planted) -> None:
    _, m, tok = planted
    lab = label_neurons(m, tok, toy.CORPUS, ["0:n0-"], llm=toy.judge())[0]
    assert lab.error and lab.recovered is None  # the switch has no negative pole on this corpus


# ---------------------------------------------------------------- diff


def test_diff_finds_the_planted_neuron(planted) -> None:
    _, m, tok = planted
    rep = diff_neurons(m, toy.finetuned(m), tok, toy.DIFF_PROMPTS)
    gained = f"{toy.GAINED[0]}:{toy.GAINED[1]}"
    assert rep.basis == "neuron"
    assert rep.top[0].unit == gained and rep.top[0].status == "gained"
    assert [c.unit for c in rep.gained] == [gained]
    assert rep.summary.change_p < 0.05
    assert rep.validity.ok and NEURON_CAVEAT in rep.caveats[0]


def test_diff_of_identical_models_is_empty(planted) -> None:
    _, m, tok = planted
    rep = diff_neurons(m, toy.model(), tok, toy.DIFF_PROMPTS)
    assert rep.summary.n_gained == rep.summary.n_lost == rep.summary.n_up == rep.summary.n_down == 0
    assert rep.summary.change_score == 0.0


def test_diff_control_marks_specific_changes(planted) -> None:
    _, m, tok = planted
    rep = diff_neurons(m, toy.finetuned(m), tok, toy.DIFF_PROMPTS, control=toy.model())
    assert rep.top[0].specific is True and rep.summary.control_specific is not None


def test_summaries_mean_stat(planted) -> None:
    _, m, tok = planted
    th = tail_thresholds(m, tok, toy.DIFF_PROMPTS, q=99)
    rates = neuron_summaries(m, tok, toy.DIFF_PROMPTS[:2], thresholds=th)
    assert all(0 < v <= 1 for g in rates for v in g.features.values())
    means = neuron_summaries(m, tok, toy.DIFF_PROMPTS[:2], stat="mean", layers=[0])
    # every layer-0 neuron except the Oakland switch, which is exactly 0 on a prompt without an O (absent = 0)
    assert set(means[0].features) == {f"0:{i}" for i in range(toy.N_NEURONS)} - {"0:1"}
    with pytest.raises(ValueError):
        neuron_summaries(m, tok, toy.DIFF_PROMPTS[:1])


# ---------------------------------------------------------------- reports and the command line


def test_reports_name_the_neuron_basis(planted, labels) -> None:
    from circuitcheck.report import render_diff, render_neuron_labels

    _, m, tok = planted
    page = render_diff(diff_neurons(m, toy.finetuned(m), tok, toy.DIFF_PROMPTS))
    assert "neurons gained" in page and "no transcoder" in page and "transcoders explain" not in page
    lab = render_neuron_labels(labels)
    assert "recovered" in lab and "letter D" in lab


def test_cli_demos(tmp_path) -> None:
    from typer.testing import CliRunner

    from circuitcheck.cli import app

    runner = CliRunner()
    r = runner.invoke(app, ["neurons", "validate", "--demo", "--json", str(tmp_path / "v.json")])
    assert r.exit_code == 0, r.output
    assert "switch" in r.output and "readout" in r.output and (tmp_path / "v.json").exists()
    r = runner.invoke(app, ["neurons", "label", "--demo", "--html", str(tmp_path / "l.html")])
    assert r.exit_code == 0, r.output
    assert "0:n0+" in r.output and "recovered  100%" in r.output
    r = runner.invoke(app, ["neurons", "diff", "--demo"])
    assert r.exit_code == 0, r.output
    assert "1:5" in r.output and "gained" in r.output
