"""Label 150 more Gemma-2-2B transcoder features (random, seed 1, layers 2-24) with Claude Haiku 5.5, detection-scored,
for the "triage hundreds of features" use case. Records from Neuronpedia; replies cached in labels_llm_cache.jsonl.

    python examples/scripts/label_gemma_batch.py
Writes examples/data/labels_gemma_batch.json.gz (records, labels, scores).
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_gemma_demo import MODEL, SOURCE, load_env

from circuitcheck.labels import FeatureData, Record, label_features, make_llm
from circuitcheck.labels.neuronpedia import fetch_feature, neuronpedia_labels, records_from_neuronpedia

DATA = Path(__file__).resolve().parents[1] / "data"
N = 150


def get(u: str):
    layer, idx = u.split(":")
    try:
        raw = fetch_feature(
            MODEL, f"{layer}-{SOURCE}", int(idx), Path.home() / ".cache" / "circuitcheck" / "neuronpedia"
        )
    except Exception:
        return None
    recs = records_from_neuronpedia(raw)
    if sum(r.peak > 0 for r in recs) < 16:
        return None
    return FeatureData(unit=u, records=recs), neuronpedia_labels(raw)


def main() -> None:
    load_env()
    rng = random.Random(1)
    cands = list(dict.fromkeys(f"{rng.randrange(2, 25)}:{rng.randrange(16384)}" for _ in range(400)))
    with ThreadPoolExecutor(8) as pool:
        got = [g for g in pool.map(get, cands) if g][:N]
    feats = [g[0] for g in got]
    theirs = {g[0].unit: g[1] for g in got}
    pool_recs = [(f.unit, r) for f in feats for r in sorted(f.records, key=lambda r: -r.peak)[:3]]
    for f in feats:
        others = [r for u, r in pool_recs if u != f.unit]
        for r in random.Random(f.unit).sample(others, 8):
            f.records.append(Record(tokens=r.tokens, acts=[0.0] * len(r.tokens), source="other-feature"))
    llm = make_llm("anthropic", "claude-haiku-5-5", DATA / "labels_llm_cache.jsonl")
    labs = label_features(llm, feats, score="detection")
    out = {
        "features": [f.model_dump() for f in feats],
        "labels": [x.model_dump() for x in labs],
        "neuronpedia_labels": theirs,
        "usage": llm.usage.__dict__,
    }
    (DATA / "labels_gemma_batch.json.gz").write_bytes(gzip.compress(json.dumps(out).encode(), mtime=0))
    ok = [x.score for x in labs if x.score is not None]
    print(f"{len(feats)} features, {len(ok)} scored, mean AUROC {sum(ok) / len(ok):.3f}, {llm.usage}")


if __name__ == "__main__":
    main()
