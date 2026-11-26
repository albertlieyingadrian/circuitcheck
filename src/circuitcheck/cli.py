"""The `circuitcheck` command: `diff`, `summarize`, `validate`, `label`, `lens`, `edges` and `version`.

circuitcheck diff graphs.json.gz --before installed/before --after installed/after --html diff.html
circuitcheck summarize graph_*.pt --group after --out graphs.json.gz
circuitcheck validate --demo --html validate.html
circuitcheck label --neuronpedia gemma-2-2b/20-gemmascope-transcoder-16k --features 15589,4012 --html labels.html
circuitcheck lens --demo --html lens.html
circuitcheck edges --demo --html edges.html
circuitcheck neurons validate --demo --html neurons.html     (also: neurons label, neurons diff)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from circuitcheck import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Diff, validate and label circuit-tracer circuits.")


def _write(obj_json: str, html_text: str, json_path: Path | None, html_path: Path | None) -> None:
    from circuitcheck.report import save_html

    if json_path:
        json_path.write_text(obj_json, encoding="utf-8")
        typer.echo(f"wrote {json_path}")
    if html_path:
        save_html(html_text, html_path)
        typer.echo(f"wrote {html_path}")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def summarize(
    graphs: Annotated[list[Path], typer.Argument(help="circuit-tracer .pt graphs or Neuronpedia graph .json files")],
    out: Annotated[Path, typer.Option(help="summary file to create or extend (.json or .json.gz)")],
    group: Annotated[str, typer.Option(help="group name for these graphs, e.g. 'before'")],
) -> None:
    """Turn graphs into the position-free summaries `diff` reads (one group per call, appended to OUT)."""
    from circuitcheck.diff.graphs import (
        GraphSummary,
        from_circuit_tracer,
        from_neuronpedia,
        load_summaries,
        save_summaries,
    )

    groups: dict[str, list[GraphSummary]] = load_summaries(out) if out.exists() else {}
    items = []
    for g in graphs:
        items.append(from_circuit_tracer(g, name=g.stem) if g.suffix == ".pt" else from_neuronpedia(g, name=g.stem))
    groups[group] = items
    save_summaries(out, groups)
    typer.echo(f"{out}: group {group!r} has {len(items)} graphs")


@app.command()
def diff(
    file: Annotated[Path, typer.Argument(help="summary file from `summarize` (or the bundled demo data)")],
    before: Annotated[str, typer.Option(help="group of graphs before training")],
    after: Annotated[str, typer.Option(help="group of graphs after training, same prompts in the same order")],
    control_before: Annotated[
        str | None, typer.Option(help="control run (did not learn the behaviour), before")
    ] = None,
    control_after: Annotated[str | None, typer.Option(help="control run, after")] = None,
    replication_before: Annotated[str | None, typer.Option(help="another seed, before")] = None,
    replication_after: Annotated[str | None, typer.Option(help="another seed, after")] = None,
    top_k: int = 10,
    min_present: int = 6,
    seed: int = 0,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Which features and edges did training add or remove? With a stability check, a null and validity checks."""
    from circuitcheck.diff.core import Settings, diff_graphs
    from circuitcheck.diff.graphs import load_summaries
    from circuitcheck.report import render_diff

    g = load_summaries(file)
    fb = g[before][0].meta.get("fvu_by_layer")
    fa = g[after][0].meta.get("fvu_by_layer")
    rep = diff_graphs(
        g[before],
        g[after],
        settings=Settings(top_k=top_k, min_present=min_present, seed=seed),
        control=(g[control_before], g[control_after]) if control_before and control_after else None,
        replication=(g[replication_before], g[replication_after]) if replication_before and replication_after else None,
        fvu_before=fb,
        fvu_after=fa,
        before_name=before,
        after_name=after,
    )
    s = rep.summary
    typer.echo(f"gained {s.n_gained}, lost {s.n_lost}, up {s.n_up}, down {s.n_down}; total change p = {s.change_p:.3f}")
    for c in rep.top:
        typer.echo(
            f"  {c.unit:>12}  {c.status:7} Δ={c.delta:+.4f}  "
            f"graphs {c.present_before}->{c.present_after}  p={c.p_null:.3f}"
        )
    typer.echo("validity: " + ("ok" if rep.validity.ok else "; ".join(rep.validity.warnings)))
    _write(rep.model_dump_json(indent=1), render_diff(rep), json_out, html)


@app.command()
def validate(
    demo: Annotated[bool, typer.Option(help="run on the built-in toy model (no GPU, no downloads)")] = False,
    model: Annotated[str | None, typer.Option(help="HF model id for circuit-tracer")] = None,
    transcoders: Annotated[str | None, typer.Option(help="transcoder set (HF id or local path)")] = None,
    prompt: Annotated[str | None, typer.Option(help="the prompt")] = None,
    counterfactual: Annotated[
        str | None, typer.Option(help="a prompt with a different answer, for the patch test")
    ] = None,
    features: Annotated[str | None, typer.Option(help="comma-separated layer:feature@pos, e.g. 20:15589@9")] = None,
    target: Annotated[str | None, typer.Option(help="answer token, e.g. ' Austin'")] = None,
    contrast: Annotated[str | None, typer.Option(help="contrast token, e.g. ' Dallas'")] = None,
    n_controls: int = 20,
    steer_scale: float = 3.0,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Is each group of features a switch or a readout? Ablation, steering and patching against controls."""
    from circuitcheck.report import render_validate
    from circuitcheck.validate import FeatureNode, logit_diff
    from circuitcheck.validate import validate as run

    if demo:
        from circuitcheck.validate.toy import demo as toy
        from circuitcheck.validate.toy import demo_nodes

        b = toy()
        metric = logit_diff(b.token_id(" Austin"), b.token_id(" Sacramento"), " Austin", " Sacramento")
        rep = run(b, "dallas", demo_nodes(b), metric=metric, counterfactual="oakland", n_controls=n_controls, seed=seed)
    else:
        if not (model and transcoders and prompt and features and target and contrast):
            raise typer.BadParameter(
                "give --demo, or all of --model --transcoders --prompt --features --target --contrast"
            )
        import torch
        from circuit_tracer import ReplacementModel

        from circuitcheck.tracer import TracerBackend

        dev = torch.device(device) if device else None
        rm = ReplacementModel.from_pretrained(model, transcoders, device=dev, dtype=torch.bfloat16)
        b2 = TracerBackend(rm)
        nodes = []
        for spec in features.split(","):
            lf, pos = spec.strip().split("@")
            layer, feat = lf.split(":")
            nodes.append(FeatureNode(layer=int(layer), pos=int(pos), feature=int(feat), activation=0.0))
        metric = logit_diff(b2.token_id(target), b2.token_id(contrast), target, contrast)
        rep = run(
            b2,
            prompt,
            nodes,
            metric=metric,
            counterfactual=counterfactual,
            n_controls=n_controls,
            steer_scale=steer_scale,
            seed=seed,
            progress=typer.echo,
        )
    for g in rep.groups:
        typer.echo(f"  {g.name:>20}  {g.verdict.value}")
    _write(rep.model_dump_json(indent=1), render_validate(rep), json_out, html)


@app.command()
def label(
    neuronpedia: Annotated[
        str | None, typer.Option(help="model/source on Neuronpedia, e.g. gemma-2-2b/20-gemmascope-transcoder-16k")
    ] = None,
    features: Annotated[str | None, typer.Option(help="comma-separated feature indices (with --neuronpedia)")] = None,
    records: Annotated[Path | None, typer.Option(help="JSON list of {unit, records:[{tokens, acts}]} instead")] = None,
    provider: Annotated[str, typer.Option(help="anthropic (Claude Haiku 5.5) or openai")] = "anthropic",
    model_name: Annotated[str | None, typer.Option("--model", help="override the provider's default model")] = None,
    score: Annotated[str, typer.Option(help="detection, simulation, both or none")] = "detection",
    compare_neuronpedia: Annotated[bool, typer.Option("--compare", help="also score Neuronpedia's labels")] = False,
    cache: Annotated[Path | None, typer.Option(help="LLM cache file (JSON lines)")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the labels as JSON here")] = None,
    export: Annotated[Path | None, typer.Option(help="write Neuronpedia-format explanation rows here")] = None,
) -> None:
    """Label features from their top activating examples and score each label on held-out examples."""
    from circuitcheck.labels import FeatureData, calibration, label_features, make_llm, score_labels
    from circuitcheck.labels.neuronpedia import compare, export_explanations, feature_data
    from circuitcheck.report import render_labels

    llm = make_llm(provider, model_name, cache)
    data: list[FeatureData] = []
    theirs: dict[str, str] = {}
    theirs_model: dict[str, str] = {}
    model_id = source = ""
    if neuronpedia:
        model_id, source = neuronpedia.split("/", 1)
        for idx in (features or "").split(","):
            fd, labs = feature_data(model_id, source, int(idx))
            data.append(fd)
            if labs:
                theirs[fd.unit], theirs_model[fd.unit] = labs[0]["description"], labs[0]["model"]
    elif records:
        data = [FeatureData.model_validate(x) for x in json.loads(records.read_text(encoding="utf-8"))]
    else:
        raise typer.BadParameter("give --neuronpedia with --features, or --records")
    ours = label_features(llm, data, score=score)
    comps = []
    if compare_neuronpedia and theirs:
        comps = compare(ours, score_labels(llm, data, theirs, score=score), theirs_model)
    for lab in ours:
        typer.echo(
            f"  {lab.unit:>10}  {lab.score if lab.score is not None else float('nan'):.2f}  {lab.label or lab.error}"
        )
    u = llm.usage
    typer.echo(
        f"API calls: {u.calls} (cache hits: {u.cached_calls}), {u.input_tokens}+{u.output_tokens} tokens, ${u.usd:.4f}"
    )
    if export and neuronpedia:
        export_explanations(ours, model_id, source, export)
    _write(
        json.dumps([x.model_dump() for x in ours], indent=1),
        render_labels(ours, calibration(ours), comps),
        json_out,
        html,
    )


DEFAULT_CONTEXTS = [
    "The museum opened in 1924 and has since grown into one of the largest collections of regional art in the "
    "country, with paintings, textiles, ceramics and photographs donated by families from across the valley. Its "
    "west wing, added after the war, holds the maps and letters of the early settlers, and every spring",
    "To repot a plant, water it the day before so the roots hold together. Choose a pot one size larger, with a "
    "drainage hole, and cover the bottom with fresh soil. Tip the plant out gently, loosen the outer roots with your "
    "fingers, set it at the same depth as before and fill around it, pressing lightly. Water again and keep",
    "The committee met on Thursday to review the budget for the coming year. Members agreed to keep spending on "
    "roads at last year's level, to raise the library's share slightly and to set aside a reserve for repairs to "
    "the school roof, which inspectors had flagged in the autumn. A final vote is expected next month after",
    "Rivers carry sediment from the mountains to the sea. Where the current slows, at a bend or where the river "
    "meets the ocean, the heavier grains settle first and the finer clay travels further. Over centuries these "
    "deposits build deltas, floodplains and sandbars, and they shift whenever a large flood changes the channel, so",
]


@app.command()
def lens(
    demo: Annotated[bool, typer.Option(help="run on the built-in toy model (no GPU, no downloads)")] = False,
    model: Annotated[str | None, typer.Option(help="HF model id for circuit-tracer")] = None,
    transcoders: Annotated[str | None, typer.Option(help="transcoder set (HF id or local path)")] = None,
    features: Annotated[
        str | None, typer.Option(help="comma-separated layer:feature[=dose], or layer:nNEURON+/- for neuron poles")
    ] = None,
    contexts: Annotated[Path | None, typer.Option(help="text file, one neutral context per line")] = None,
    r_lens: Annotated[
        str | None, typer.Option(help="R-lens .pt file, or 'hub' for neuron-lens's release (Qwen3, Gemma 4)")
    ] = None,
    n_random: int = 16,
    dose_scale: float = 1.0,
    n_boot: int = 2000,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Does each unit's lens read predict what steering it does? Against random units of the same layer and norm."""
    from circuitcheck.lens import LensUnit, lens_predicts_steering
    from circuitcheck.report import render_lens

    if demo:
        from circuitcheck.lens.toy import demo as toy
        from circuitcheck.lens.toy import demo_units

        rep = lens_predicts_steering(
            toy(), demo_units(), n_random=n_random, dose_scale=dose_scale, n_boot=n_boot, seed=seed, label_top=3
        )
    else:
        if not (model and transcoders and features):
            raise typer.BadParameter("give --demo, or all of --model --transcoders --features")
        import torch
        from circuit_tracer import ReplacementModel

        from circuitcheck.tracer import TracerLensBackend

        units = []
        for spec in features.split(","):
            spec, _, dose = spec.strip().partition("=")
            layer, idx = spec.split(":")
            if idx.startswith("n"):
                sign = -1 if idx.endswith("-") else 1
                units.append(LensUnit(layer=int(layer), index=int(idx[1:].rstrip("+-")), kind="neuron", sign=sign))
            else:
                units.append(LensUnit(layer=int(layer), index=int(idx), dose=float(dose) if dose else None))
        dev = torch.device(device) if device else None
        rm = ReplacementModel.from_pretrained(model, transcoders, device=dev, dtype=torch.float32)
        if r_lens == "hub":
            from circuitcheck.lens.neuron_lens import r_lens_path

            r_lens = str(r_lens_path(model))
        texts = contexts.read_text(encoding="utf-8").splitlines() if contexts else DEFAULT_CONTEXTS
        backend = TracerLensBackend(rm, [t for t in texts if t.strip()], r_lens=r_lens)
        rep = lens_predicts_steering(
            backend, units, n_random=n_random, dose_scale=dose_scale, n_boot=n_boot, seed=seed, progress=typer.echo
        )
    for s in rep.summary.values():
        typer.echo(f"  {s.verdict}; difference {s.diff.value:+.2f} [{s.diff.low:+.2f}, {s.diff.high:+.2f}]")
    for u in rep.flagged():
        typer.echo(f"  flagged {u.unit.name()}: " + "; ".join(u.flags))
    _write(rep.model_dump_json(indent=1), render_lens(rep), json_out, html)


@app.command()
def edges(
    file: Annotated[Path | None, typer.Argument(help="summary file from `summarize`")] = None,
    demo: Annotated[bool, typer.Option(help="run on a toy catalog and graph (no downloads)")] = False,
    group: Annotated[str | None, typer.Option(help="group of graphs in FILE; their edges are pooled")] = None,
    catalog: Annotated[Path | None, typer.Option(help="edge catalog JSON (EdgeCatalog.save)")] = None,
    neuron_lens: Annotated[
        str | None, typer.Option(help="download neuron-lens's catalog for this model id, e.g. Qwen/Qwen3-0.6B")
    ] = None,
    table: Annotated[str, typer.Option(help="edges (weight wiring) or sources (drivers at top firings)")] = "edges",
    unit_map: Annotated[
        Path | None, typer.Option(help="JSON {unit: [[layer, neuron, sign], ...]} for feature graphs")
    ] = None,
    top: Annotated[int | None, typer.Option(help="keep the strongest edges by |weight|")] = None,
    n_null: int = 200,
    seed: int = 0,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """What share of a graph's edges does a whole-model edge catalog list? Against degree-matched rewirings."""
    from circuitcheck.lens import EdgeCatalog, edges_agree
    from circuitcheck.report import render_edges

    mp = None
    if demo:
        from circuitcheck.lens.toy import demo_edges

        g, cat = demo_edges(seed)
    else:
        if not (file and group and (catalog or neuron_lens)):
            raise typer.BadParameter("give --demo, or FILE --group with --catalog or --neuron-lens")
        from circuitcheck.diff.graphs import load_summaries, parse_unit

        g = {}
        for summ in load_summaries(file)[group]:
            for k, w in summ.edges.items():
                g[k] = g.get(k, 0.0) + w
        if catalog:
            cat = EdgeCatalog.load(catalog)
        else:
            from circuitcheck.lens.neuron_lens import load_catalog

            layers = {parse_unit(u)[0] for e in g for u in e}
            cat = load_catalog(str(neuron_lens), layers, table=table)
        if unit_map:
            raw = json.loads(unit_map.read_text(encoding="utf-8"))
            mp = {u: [(int(a), int(b), int(c)) for a, b, c in ps] for u, ps in raw.items()}
    rep = edges_agree(g, cat, unit_map=mp, top=top, n_null=n_null, seed=seed)
    typer.echo(f"  {rep.verdict}")
    typer.echo(f"  compared {rep.n_compared} of {rep.n_edges} edges; skipped {rep.skipped}")
    _write(rep.model_dump_json(indent=1), render_edges(rep), json_out, html)


neurons_app = typer.Typer(
    no_args_is_help=True,
    help="validate, label and diff on raw MLP neurons (no transcoders), through neuronkit (install the neurons extra).",
)
app.add_typer(neurons_app, name="neurons")


def _hf(model: str, device: str | None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m: Any = AutoModelForCausalLM.from_pretrained(model)
    m.eval()
    if device:
        m = m.to(device)
    return m, AutoTokenizer.from_pretrained(model)


def _lines(path: Path) -> list[str]:
    return [t for t in path.read_text(encoding="utf-8").splitlines() if t.strip()]


@neurons_app.command("validate")
def neurons_validate(
    demo: Annotated[bool, typer.Option(help="run on the planted toy transformer (CPU, no downloads)")] = False,
    model: Annotated[str | None, typer.Option(help="HF model id")] = None,
    prompt: Annotated[str | None, typer.Option(help="the prompt")] = None,
    counterfactual: Annotated[
        str | None, typer.Option(help="a prompt with a different answer, for the patch test")
    ] = None,
    neurons: Annotated[
        str | None, typer.Option(help="comma-separated layer:neuron@pos, e.g. 12:345@-1 (-1 = last token)")
    ] = None,
    target: Annotated[str | None, typer.Option(help="answer token, e.g. ' Austin'")] = None,
    contrast: Annotated[str | None, typer.Option(help="contrast token, e.g. ' Dallas'")] = None,
    n_controls: int = 20,
    steer_scale: float = 3.0,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Is each group of neurons a switch or a readout? Ablation, steering and patching against neuron controls."""
    from circuitcheck.neurons import NeuronBackend
    from circuitcheck.report import render_validate
    from circuitcheck.validate import logit_diff
    from circuitcheck.validate import validate as run

    if demo:
        from circuitcheck.neurons import toy

        b, _, _ = toy.demo()
        prompt, counterfactual = toy.PROMPT, toy.COUNTERFACTUAL
        target, contrast, specs = toy.TARGET, toy.CONTRAST, toy.demo_specs()
    else:
        if not (model and prompt and neurons and target and contrast):
            raise typer.BadParameter("give --demo, or all of --model --prompt --neurons --target --contrast")
        m, tok = _hf(model, device)
        b = NeuronBackend(m, tok)
        specs = [s.strip() for s in neurons.split(",")]
    nodes = b.nodes(prompt, specs)
    metric = logit_diff(b.token_id(target), b.token_id(contrast), target, contrast)
    rep = run(
        b,
        prompt,
        nodes,
        metric=metric,
        counterfactual=counterfactual,
        n_controls=n_controls,
        steer_scale=steer_scale,
        seed=seed,
        progress=None if demo else typer.echo,
    )
    for g in rep.groups:
        typer.echo(f"  {g.name:>20}  {g.verdict.value}")
    _write(rep.model_dump_json(indent=1), render_validate(rep, title="Neuron circuit validation"), json_out, html)


@neurons_app.command("label")
def neurons_label(
    demo: Annotated[
        bool, typer.Option(help="planted toy transformer and a scripted judge (no downloads, no key)")
    ] = False,
    model: Annotated[str | None, typer.Option(help="HF model id")] = None,
    neurons: Annotated[str | None, typer.Option(help="comma-separated poles, e.g. 12:n345+,14:n880-")] = None,
    corpus: Annotated[Path | None, typer.Option(help="text file, one document per line")] = None,
    descriptions: Annotated[
        Path | None, typer.Option(help="JSON {pole: description} to test instead of writing new ones")
    ] = None,
    provider: Annotated[str, typer.Option(help="anthropic (Claude Haiku 5.5) or openai")] = "anthropic",
    llm_model: Annotated[str | None, typer.Option(help="override the provider's default model")] = None,
    cache: Annotated[Path | None, typer.Option(help="LLM cache file (JSON lines)")] = None,
    n_explain: int = 12,
    n_write: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the labels as JSON here")] = None,
) -> None:
    """Describe neuron poles and test each description: predict, simulate (ablation damage recovered) and write."""
    from circuitcheck.neurons import label_neurons
    from circuitcheck.report import render_neuron_labels

    given = json.loads(descriptions.read_text(encoding="utf-8")) if descriptions else None
    m: Any
    tok: Any
    llm: Any
    if demo:
        from circuitcheck.neurons import toy

        m, tok, llm = toy.model(), toy.CharTokenizer(), toy.judge()
        texts, poles = toy.CORPUS, ["0:n0+", "1:n0+"]
        n_explain = min(n_explain, 4)
    else:
        if not (model and neurons and corpus):
            raise typer.BadParameter("give --demo, or all of --model --neurons --corpus")
        from circuitcheck.labels import make_llm

        m, tok = _hf(model, device)
        llm = make_llm(provider, llm_model, cache)
        texts, poles = _lines(corpus), [p.strip() for p in neurons.split(",")]
    labs = label_neurons(m, tok, texts, poles, llm=llm, descriptions=given, n_explain=n_explain, n_write=n_write)
    for lab in labs:
        rec = "  n/a" if lab.recovered is None else f"{lab.recovered:5.0%}"
        acc = "n/a" if lab.predict_accuracy is None else f"{lab.predict_accuracy:.2f}"
        typer.echo(f"  {lab.unit:>10}  recovered {rec}  predict {acc}  {lab.label or lab.error}")
    _write(json.dumps([x.model_dump() for x in labs], indent=1), render_neuron_labels(labs), json_out, html)


@neurons_app.command("diff")
def neurons_diff(
    demo: Annotated[bool, typer.Option(help="the toy transformer against a copy with one planted change")] = False,
    before: Annotated[str | None, typer.Option(help="HF model id or path before training")] = None,
    after: Annotated[str | None, typer.Option(help="HF model id or path after training (same tokenizer)")] = None,
    control: Annotated[str | None, typer.Option(help="a control run's model (did not learn the behaviour)")] = None,
    prompts: Annotated[Path | None, typer.Option(help="text file, one prompt per line")] = None,
    layers: Annotated[str | None, typer.Option(help="comma-separated layers (default: all)")] = None,
    stat: Annotated[str, typer.Option(help="tail_rate or mean")] = "tail_rate",
    q: Annotated[float, typer.Option(help="tail percentile from the before model (p q and p 100-q)")] = 99.0,
    top_k: int = 10,
    min_present: int = 6,
    seed: int = 0,
    device: Annotated[str | None, typer.Option(help="cuda, mps or cpu")] = None,
    html: Annotated[Path | None, typer.Option(help="write the HTML report here")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="write the JSON report here")] = None,
) -> None:
    """Which neurons fire more or less in their tails after training? Paired swap null, BH-corrected."""
    from circuitcheck.diff.core import Settings
    from circuitcheck.neurons import diff_neurons
    from circuitcheck.report import render_diff

    if stat not in ("tail_rate", "mean"):
        raise typer.BadParameter("--stat is tail_rate or mean")
    ctrl = None
    mb: Any
    tok: Any
    if demo:
        from circuitcheck.neurons import toy

        mb, tok = toy.model(), toy.CharTokenizer()
        ma, texts = toy.finetuned(mb), toy.DIFF_PROMPTS
    else:
        if not (before and after and prompts):
            raise typer.BadParameter("give --demo, or all of --before --after --prompts")
        mb, tok = _hf(before, device)
        ma, _ = _hf(after, device)
        ctrl = _hf(control, device)[0] if control else None
        texts = _lines(prompts)
    rep = diff_neurons(
        mb,
        ma,
        tok,
        texts,
        stat=stat,  # type: ignore[arg-type]
        q=q,
        layers=[int(x) for x in layers.split(",")] if layers else None,
        control=ctrl,
        settings=Settings(top_k=top_k, min_present=min_present, seed=seed),
    )
    s = rep.summary
    typer.echo(f"gained {s.n_gained}, lost {s.n_lost}, up {s.n_up}, down {s.n_down}; total change p = {s.change_p:.3f}")
    for c in rep.top:
        typer.echo(
            f"  {c.unit:>12}  {c.status:7} Δ={c.delta:+.4f}  prompts {c.present_before}->{c.present_after}  "
            f"p={c.p_null:.3f}"
        )
    _write(rep.model_dump_json(indent=1), render_diff(rep, title="Neuron diff"), json_out, html)


if __name__ == "__main__":  # pragma: no cover
    app()
