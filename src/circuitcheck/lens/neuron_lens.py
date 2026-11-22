"""Read neuron-lens's public catalog and lenses from the Hugging Face Hub.

neuron-lens (github.com/stdlogits/neuron-lens, MIT) publishes a per-neuron catalog in the `stdlogits/neuron-lens`
dataset and fitted Jacobian lenses in the `stdlogits/rs-lenses` model repo, for Qwen3 0.6B-14B (and their base
models) and Gemma 4 12B. Needs `huggingface_hub` and `pyarrow` (`pip install "circuitcheck[neuronlens]"`); files are
cached by the Hub client.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from circuitcheck.lens.edges import EdgeCatalog

CATALOG_REPO = "stdlogits/neuron-lens"
LENS_REPO = "stdlogits/rs-lenses"

# Hugging Face model id -> folder in both repos
COVERED: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-0.6B-Base": "qwen3-0.6b-base",
    "Qwen/Qwen3-1.7B": "qwen3-1.7b",
    "Qwen/Qwen3-1.7B-Base": "qwen3-1.7b-base",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-4B-Base": "qwen3-4b-base",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-8B-Base": "qwen3-8b-base",
    "Qwen/Qwen3-14B": "qwen3-14b",
    "Qwen/Qwen3-14B-Base": "qwen3-14b-base",
    "google/gemma-4-12B": "gemma-4-12b",
}


def folder(model_id: str) -> str:
    """The catalog folder for a model id, or a ValueError naming the covered models."""
    key = next((k for k in COVERED if k.lower() == model_id.lower()), None)
    if key is not None:
        return COVERED[key]
    if model_id in COVERED.values():
        return model_id
    raise ValueError(f"neuron-lens does not cover {model_id!r}; covered: {', '.join(COVERED)}")


def is_covered(model_id: str) -> bool:
    try:
        folder(model_id)
    except ValueError:
        return False
    return True


def _download(repo: str, path: str, repo_type: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type=repo_type))


def load_catalog(model_id: str, layers: Iterable[int], table: str = "edges") -> EdgeCatalog:
    """The `edges` (weight wiring) or `sources` (drivers at top firings) table of the given layers as an
    `EdgeCatalog`. Load every layer a graph touches: an edge is checked only when both of its layers are loaded."""
    import pyarrow.parquet as pq

    if table not in ("edges", "sources"):
        raise ValueError(f"table must be 'edges' or 'sources', not {table!r}")
    f = folder(model_id)
    cols = (
        ["layer", "neuron", "pole", *(f"{s}_{c}" for s in ("in", "out") for c in ("layer", "neuron", "pole", "push"))]
        if table == "edges"
        else ["layer", "neuron", "pole", "driven_layer", "driven_neuron", "driven_pole", "driven_share"]
    )
    rows: list[dict[str, object]] = []
    loaded: list[int] = []
    for layer in sorted(set(layers)):
        loaded.append(layer)
        if table == "sources" and layer == 0:  # layer 0 has no earlier poles, so no sources file
            continue
        p = _download(CATALOG_REPO, f"{f}/neurons/{table}/L{layer:02d}.parquet", "dataset")
        rows += pq.read_table(p, columns=cols).to_pylist()
    cat = EdgeCatalog.from_rows(rows, table=table, name=f"neuron-lens {f}")
    cat.layers |= set(loaded)
    return cat


def labels(model_id: str, layer: int) -> dict[tuple[int, int], str]:
    """neuron-lens's 'writes' label per (neuron, pole) of one layer (LLM labels written from the catalog evidence)."""
    import pyarrow.parquet as pq

    p = _download(CATALOG_REPO, f"{folder(model_id)}/neurons/labels/L{layer:02d}.parquet", "dataset")
    t = pq.read_table(p, columns=["neuron", "pole", "writes"]).to_pylist()
    return {(int(r["neuron"]), int(r["pole"])): str(r["writes"]) for r in t if r["writes"]}


def r_lens_path(model_id: str) -> Path:
    """Local path of the released R-lens (layer L's output -> the last layer's output, `{"J": [L, d, d]}`)."""
    return _download(LENS_REPO, f"{folder(model_id)}/r-lens/lens.pt", "model")


def token_counts_path(model_id: str) -> Path:
    """Local path of the catalog's token counts over its 15M-token corpus (to keep tokens seen 5+ times)."""
    return _download(CATALOG_REPO, f"{folder(model_id)}/token_counts.pt", "dataset")
