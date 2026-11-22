"""Run the lens checks on a real model (Modal GPU) and save what the tests and docs read.

On Qwen3-0.6B (`mwhanna/qwen3-0.6b-transcoders-lowl0`) or Qwen3-4B (`mwhanna/qwen3-4b-transcoders`, whose features
have Neuronpedia labels), both covered by neuron-lens's catalog and lenses:

1. Attribution graphs for three prompts (circuit-tracer), summarised with `from_circuit_tracer`; the 30 most
   influential features of each.
2. `lens_predicts_steering` on those features (logit lens and neuron-lens's R-lens), against 16 random features a
   layer at the same write norm, steered in 16 WikiText paragraphs of 64 tokens. Neuronpedia labels (Qwen3-4B) are
   checked against the reads.
3. The same check on random MLP neuron poles of every third layer, with neuron-lens's own labels: a replication of
   neuron-lens's findings §3 on this model.
4. `edges_agree`: each graph's feature-to-feature edges, mapped to neuron poles by decoder-vs-`W_out` cosine (the
   nearest 1 and 3 poles), against neuron-lens's `edges` and `sources` tables, with 200 degree-matched rewirings.

Needs a Modal secret `huggingface-secret` (an HF token, for download rate limits).

    MODAL_PROFILE=... python -m modal run examples/modal/lens_checks.py --job qwen3-0.6b
    MODAL_PROFILE=... python -m modal run examples/modal/lens_checks.py --job qwen3-4b

Output: `examples/data/live/lens_<job>.json`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL = len(HERE.parents) >= 3  # False inside the container
PKG = HERE.parents[1] if LOCAL else HERE
CT_COMMIT = "720f24e5939cab0642daf324eec7d3ce57e01aa1"

JOBS = {
    "qwen3-0.6b": {
        "model": "Qwen/Qwen3-0.6B",
        "transcoders": "mwhanna/qwen3-0.6b-transcoders-lowl0",
        "dtype": "float32",
        "neuronpedia": None,
        "batch": 8,
    },
    "qwen3-4b": {
        "model": "Qwen/Qwen3-4B",
        "transcoders": "mwhanna/qwen3-4b-transcoders",
        "dtype": "bfloat16",
        "neuronpedia": ("qwen3-4b", "transcoder-hp"),
        "batch": 4,
    },
}
PROMPTS = [
    "Fact: the capital of the state containing Dallas is",
    'The opposite of "hot" is "',
    "When Mary and John went to the store, John gave a drink to",
]

app = modal.App("circuitcheck-lens")
hf_vol = modal.Volume.from_name("circuitcheck-hf", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.3",
        "transformer-lens==3.2.1",
        f"circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer@{CT_COMMIT}",
        "huggingface_hub[hf_transfer]<1.0",
        "pyarrow>=15",
        "numpy<2.3",
        "pydantic>=2.7",
        "typer>=0.12",
        "requests",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "PYTHONPATH": "/root/ck"})
)
if LOCAL:
    image = image.add_local_dir(PKG / "src" / "circuitcheck", "/root/ck/circuitcheck", ignore=["**/__pycache__/**"])


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def wikitext(tok, n: int = 16, length: int = 64) -> list[str]:
    """The first `n` WikiText-103 test paragraphs with at least `length` tokens, as neutral steering contexts."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    out = []
    for text in pq.read_table(p).column("text").to_pylist():
        t = text.strip()
        if t and not t.startswith("=") and len(tok(t, add_special_tokens=False)["input_ids"]) >= length:
            out.append(t)
        if len(out) == n:
            break
    return out


def neuronpedia_labels(model_id: str, source: str, units) -> dict[tuple[int, int], str]:
    import requests

    out = {}
    for u in units:
        try:
            url = f"https://www.neuronpedia.org/api/feature/{model_id}/{u.layer}-{source}/{u.index}"
            ex = requests.get(url, timeout=30).json().get("explanations") or []
            if ex and ex[0].get("description"):
                out[(u.layer, u.index)] = str(ex[0]["description"]).strip()
        except Exception as err:  # a missing label is not fatal
            log(f"neuronpedia {u.layer}/{u.index}: {err}")
    return out


@app.function(
    image=image,
    gpu=["H100", "A100-80GB"],
    volumes={"/hf": hf_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3 * 3600,
    memory=96 * 1024,
)
def run(job: str, n_random: int = 16, top: int = 30, neurons_per_layer: int = 4) -> str:
    import random

    import numpy as np
    import torch
    from circuit_tracer import ReplacementModel, attribute

    from circuitcheck.diff.graphs import from_circuit_tracer
    from circuitcheck.lens import LensUnit, edges_agree, lens_predicts_steering, units_from_nodes
    from circuitcheck.lens import neuron_lens as nl
    from circuitcheck.tracer import TracerLensBackend, map_features_to_neurons, top_nodes

    cfg = JOBS[job]
    t0 = time.time()
    dtype = getattr(torch, cfg["dtype"])
    model = ReplacementModel.from_pretrained(cfg["model"], cfg["transcoders"], device=torch.device("cuda"), dtype=dtype)
    hf_vol.commit()
    log(f"loaded {cfg['model']} in {time.time() - t0:.0f}s")
    out: dict = {"job": job, "config": {k: v for k, v in cfg.items() if k != "neuronpedia"}, "prompts": PROMPTS}

    # 1. graphs
    summaries, nodes = [], []
    out["graphs"] = []
    for i, prompt in enumerate(PROMPTS):
        graph = attribute(
            prompt, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=128, max_feature_nodes=4096
        )
        s = from_circuit_tracer(graph, name=f"prompt_{i}")
        ns = top_nodes(graph, top)
        out["graphs"].append(
            {"name": s.name, "prompt": prompt, "replacement": s.replacement_score,
             "completeness": s.completeness_score, "n_features": len(s.features), "n_edges": len(s.edges),
             "top": [n.name() for n in ns[:10]]}
        )  # fmt: skip
        summaries.append(s)
        nodes += ns
        log(f"graph {i}: {len(s.features)} units, {len(s.edges)} edges, replacement {s.replacement_score:.2f}")
        del graph
        torch.cuda.empty_cache()

    # 2. lens check on graph features
    units = units_from_nodes(nodes)
    if cfg["neuronpedia"]:
        labs = neuronpedia_labels(*cfg["neuronpedia"], units)
        units = [u.model_copy(update={"label": labs.get((u.layer, u.index))}) for u in units]
        out["labelled"] = len(labs)
    counts = torch.load(nl.token_counts_path(cfg["model"]), weights_only=False)["counts"]
    keep = (counts >= 5).numpy()
    mask = np.zeros(int(model.cfg.d_vocab), dtype=bool)
    mask[: min(len(keep), len(mask))] = keep[: len(mask)]
    contexts = wikitext(model.tokenizer)
    backend = TracerLensBackend(
        model, contexts, r_lens=nl.r_lens_path(cfg["model"]), vocab_mask=mask, batch_size=cfg["batch"]
    )
    log(f"lens backend: {backend.describe()}; {len(units)} feature units")
    rep = lens_predicts_steering(backend, units, n_random=n_random, seed=0, progress=log)
    out["lens_features"] = json.loads(rep.model_dump_json())
    for s in rep.summary.values():
        log(s.verdict)

    # 3. random neuron poles with neuron-lens's labels, steered at 4x a unit activation
    rng = random.Random(0)
    n_layers, d_mlp = int(model.cfg.n_layers), int(model.cfg.d_mlp)
    poles = []
    for layer in range(1, n_layers, 3):
        labs = nl.labels(cfg["model"], layer)
        for n in rng.sample(range(d_mlp), neurons_per_layer):
            sign = rng.choice((1, -1))
            poles.append(LensUnit(layer=layer, index=n, kind="neuron", sign=sign, label=labs.get((n, sign))))
    rep_n = lens_predicts_steering(backend, poles, n_random=n_random, dose_scale=4.0, seed=1, progress=log)
    out["lens_neurons"] = json.loads(rep_n.model_dump_json())
    for s in rep_n.summary.values():
        log("neurons: " + s.verdict)

    # 4. graph edges against the catalog
    out["edges"] = {}
    feats = sorted({u for s in summaries for e in s.edges for u in e})
    layers = sorted({int(u.split(":")[0]) for u in feats})
    cats = {t: nl.load_catalog(cfg["model"], layers, table=t) for t in ("edges", "sources")}
    pooled: dict = {}
    for s in summaries:
        for e, w in s.edges.items():
            pooled[e] = pooled.get(e, 0.0) + w
    for k in (1, 3):
        mp, cos = map_features_to_neurons(model, feats, k=k)
        best = [cs[-1] for cs in cos.values()]  # the k-th nearest pole's |cosine|
        out.setdefault("map_cos", {})[f"k{k}"] = {
            "median": float(np.median(best)), "q25": float(np.quantile(best, 0.25)),
            "q75": float(np.quantile(best, 0.75)), "one_over_sqrt_d": float(1 / np.sqrt(model.cfg.d_model)),
        }  # fmt: skip
        for t, cat in cats.items():
            for s in summaries:
                r = edges_agree(s.edges, cat, unit_map=mp, n_null=200, seed=0)
                out["edges"][f"{s.name}/k{k}/{t}"] = json.loads(r.model_dump_json())
            r = edges_agree(pooled, cat, unit_map=mp, n_null=200, seed=0)
            out["edges"][f"pooled/k{k}/{t}"] = json.loads(r.model_dump_json())
            log(f"edges k={k} {t}: {r.verdict}")
    out["seconds"] = time.time() - t0
    return json.dumps(out)


@app.local_entrypoint()
def main(job: str = "qwen3-0.6b", n_random: int = 16) -> None:
    res = run.remote(job, n_random=n_random)
    dest = PKG / "examples" / "data" / "live" / f"lens_{job}.json"
    dest.write_text(json.dumps(json.loads(res), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {dest}")
