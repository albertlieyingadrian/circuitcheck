"""Label the circuits episodes' own transcoder features (Qwen3-1.7B RL hack, Qwen3-8B password lock) with Claude
Haiku 5.5, and score the episodes' Claude Opus 5.5 labels with the same detector.

Records come from each episode's `results/explain_acts.npz` (every unit's activation on every token of the texts
the episode's explainer saw). The 10 claimed features and 10 random features of each episode are labelled.

Writes, in examples/data/: `episode_features.json.gz` (records), `episode_labels.json` (labels, scores, Opus
comparison) and adds every reply to `labels_llm_cache.jsonl`, so reruns and the docs run without a key.

    python examples/scripts/label_episode_features.py <steerlab checkout>
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

from circuitcheck.labels import FeatureData, Record, calibration, label_features, make_llm, score_labels

DATA = Path(__file__).resolve().parents[1] / "data"
EPISODES = {"rl-hack-circuit": "Qwen3-1.7B after Pigment Gym RL", "lock-circuit": "password-locked Qwen3-8B"}


def load_env() -> None:
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / ".env").exists():
            for line in (d / ".env").read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def features(root: Path, episode: str) -> tuple[list[FeatureData], dict[str, str]]:
    res = root / "experiments" / episode / "results"
    meta = json.loads((res / "explain_meta.json").read_text())
    acts = np.load(res / "explain_acts.npz")
    rubrics = json.loads((res / "rubrics.json").read_text()) if (res / "rubrics.json").exists() else {}
    out, opus = [], {}
    for j, u in enumerate(meta["units"]):
        if u["kind"] != "feature":
            continue
        recs = [
            Record(tokens=toks, acts=[max(0.0, float(x)) for x in acts[key][j]], source=key)
            for key, toks in meta["tokens"].items()
            if key in acts
        ]
        unit = f"{u['layer']}:{u['index']}"
        out.append(FeatureData(unit=unit, records=recs))
        if u["id"] in rubrics:
            opus[unit] = rubrics[u["id"]]["label"]
    return out, opus


def main(root: Path) -> None:
    load_env()
    llm = make_llm("anthropic", "claude-haiku-5-5", DATA / "labels_llm_cache.jsonl")
    payload, results = {}, {}
    for ep, desc in EPISODES.items():
        feats, opus = features(root, ep)
        payload[ep] = [f.model_dump() for f in feats]
        ours = label_features(llm, feats, score="both")
        theirs = score_labels(llm, feats, opus, score="both")
        results[ep] = {
            "model": desc,
            "ours": [x.model_dump() for x in ours],
            "opus": [x.model_dump() for x in theirs],
            "calibration": calibration(ours).model_dump(),
        }
        ok = [x.score for x in ours if x.score is not None]
        print(f"{ep}: {len(feats)} features, mean score {np.mean(ok):.3f}, opus labels scored: {len(theirs)}")
    (DATA / "episode_features.json.gz").write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    results["usage"] = llm.usage.__dict__
    (DATA / "episode_labels.json").write_text(json.dumps(results, indent=1) + "\n")
    print(llm.usage)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
