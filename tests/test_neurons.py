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
    # every layer-0 neuron except the Oakland switch and the T detector, exactly 0 on a prompt without an O or a T
    assert set(means[0].features) == {f"0:{i}" for i in range(toy.N_NEURONS)} - {"0:1", "0:2"}
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


# ---------------------------------------------------------------- lens


@pytest.fixture(scope="module")
def lens_backend(planted):
    from circuitcheck.neurons import NeuronLensBackend

    _, m, tok = planted
    return NeuronLensBackend(m, tok, toy.LENS_CONTEXTS, ctx_len=48, skip_first=4)


@pytest.fixture(scope="module")
def lens_report(lens_backend):
    from circuitcheck.neurons import neuron_lens

    return neuron_lens(lens_backend, toy.lens_demo_poles(), n_random=16, n_boot=500)


def _unit(report, name: str):
    return next(u for u in report.units if u.unit.name() == name)


def test_lens_reads_the_switch_as_its_answer(lens_report) -> None:
    sw = _unit(lens_report, "L0/n0+")
    for lens in ("logit", "jacobian"):
        assert sw.scores[lens].top[0] == toy.TARGET and sw.scores[lens].rho > 0.9
    assert sw.steer_top[0] == toy.TARGET and not sw.flags


def test_jacobian_lens_sees_through_the_relay(lens_report) -> None:
    """The relay source writes a direction the unembedding cannot see: the logit lens reads nothing and is flagged,
    while the Jacobian lens reads what its steering does (A, through L1/N7)."""
    relay = _unit(lens_report, "L0/n2+")
    assert abs(relay.scores["logit"].rho) < 0.1 and any("lens misses (logit)" in f for f in relay.flags)
    assert relay.scores["jacobian"].top[0] == toy.TARGET and relay.scores["jacobian"].rho > 0.9
    assert relay.steer_top[0] == toy.TARGET


def test_gate_closed_poles_are_flagged(lens_report, lens_backend) -> None:
    neg = _unit(lens_report, "L0/n0-")
    assert any(f.startswith("gate closed") for f in neg.flags)
    assert lens_backend.open_rate(0, 0, +1) > 0 and lens_backend.open_rate(0, 0, -1) == 0
    assert any("gate closed" in r for r in lens_report.rules)


def test_logit_lens_folds_in_the_final_norm_gain(lens_backend, planted) -> None:
    from circuitcheck.lens import LensUnit

    _, m, _ = planted
    u = LensUnit(layer=0, index=3, kind="neuron", sign=-1)
    w = lens_backend.write_vectors([u])
    want = (m.lm_head.weight.detach() * m.model.norm.weight.detach()[None, :]).numpy() @ w[0]
    np.testing.assert_allclose(lens_backend.lens_logits(0, w, "logit")[0], want, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(w[0], -nk.sites(m)[0].down_cols()[3].detach().numpy(), rtol=1e-6)


def test_jacobian_fallback_matches_neuronkit(lens_backend) -> None:
    """The exact JVP (used when neuronkit has no readout module) agrees with neuronkit's finite difference."""
    readout = pytest.importorskip("neuronkit.readout")
    v = torch.tensor(lens_backend.write_vectors([_pole(toy.RELAY_SOURCE)])[0])
    ours = lens_backend.jacobian_jvp(0, v)
    theirs = readout.jacobian_direction(lens_backend.model, lens_backend.tokenizer, 0, v, lens_backend.texts)
    assert torch.nn.functional.cosine_similarity(ours, theirs.float(), dim=0) > 0.999
    # the relay carries the write on to the Austin direction to first order, not only through a second-order term
    assert abs(float(ours[toy.AUSTIN_DIM])) > 0.05
    assert torch.allclose(ours, theirs.float(), rtol=1e-2, atol=1e-3)


def test_gate_fallback_matches_neuronkit(planted) -> None:
    from circuitcheck.neurons import NeuronLensBackend

    _, m, tok = planted
    b = NeuronLensBackend(m, tok, toy.LENS_CONTEXTS, ctx_len=48, skip_first=4)
    if getattr(nk, "gate_open", None) is not None:
        b._open.clear()
        with_nk = [b.open_rate(0, i, s) for i in range(4) for s in (1, -1)]
        g = b._gate_preacts(0)
        assert torch.equal(b._open[0][1].bool(), (g > 0).to(b._open[0][1].device))
        assert with_nk[0] > 0


def _pole(unit: tuple[int, int], sign: int = 1):
    from circuitcheck.lens import LensUnit

    return LensUnit(layer=unit[0], index=unit[1], kind="neuron", sign=sign)


def test_cli_lens_demo() -> None:
    from typer.testing import CliRunner

    from circuitcheck.cli import app

    r = CliRunner().invoke(app, ["neurons", "lens", "--demo", "--n-boot", "200"])
    assert r.exit_code == 0, r.output
    assert "L0/n2+" in r.output and "gate closed" in r.output and "lens misses (logit)" in r.output
