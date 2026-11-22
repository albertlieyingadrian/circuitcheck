---
name: circuitcheck
description: "Run this repo's `circuitcheck` toolkit on circuit-tracer attribution graphs: validate whether graph features are a switch or only a readout (ablation, steering and patching against random and size-matched controls), diff graphs before and after fine-tuning or RL (swap null, stability, control run, reconstruction-error validity check), and label features cheaply with held-out trust scores. Use when the user asks whether a circuit or feature causes a behaviour, what training changed in a model's circuits, to compare checkpoints, to label transcoder features, or to read a circuitcheck HTML/JSON report."
allowed-tools: Bash(.venv/bin/circuitcheck --help), Bash(.venv/bin/circuitcheck version), Bash(.venv/bin/circuitcheck validate --demo:*), Bash(.venv/bin/circuitcheck diff examples/data/*)
user-invocable: true
---

# circuitcheck

The CLI is `.venv/bin/circuitcheck` once installed: `pip install circuitcheck` from PyPI, or `uv venv && uv pip
install -e ".[dev]"` in a clone (the example data lives in the clone). Every
command prints `--help`. The base install has no torch. The toy model, the bundled graphs in `examples/data/` and the
cached model replies let every tool run offline. Real models need circuit-tracer from GitHub and a GPU.

## Pick the tool from the question

| The user asks | Tool | Command |
|---|---|---|
| "Does this feature / circuit cause X?" "Is the graph right?" | validate | `circuitcheck validate` |
| "What did fine-tuning / RL change?" "Compare checkpoints" | diff | `circuitcheck summarize`, then `circuitcheck diff` |
| "What does feature L:F do?" "Label these features" | labels | `circuitcheck label` |
| "Does the lens read predict steering?" "Is this label right?" | lens | `circuitcheck lens` |
| "Do the graph's edges match the model's wiring?" | edges | `circuitcheck edges` |

## Start here (offline, instant, pre-approved)

```bash
.venv/bin/circuitcheck validate --demo                     # toy model: one switch, two readouts, a backup pair
.venv/bin/circuitcheck diff examples/data/rl_hack_graphs.json.gz \
  --before installed/before --after installed/after \
  --control-before control/before --control-after control/after
```

Add `--html report.html --json report.json` to any command for the reports.

## Validate (a real model loads on a GPU: ask first)

```bash
.venv/bin/circuitcheck validate --model google/gemma-2-2b --transcoders gemma \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --counterfactual "Fact: the capital of the state containing Oakland is" \
  --features 20:15589@9,16:25@9 --target " Austin" --contrast " Sacramento" --json out.json
```

Features are `layer:feature@position`. In Python, `circuitcheck.tracer.TracerBackend(model)` wraps a circuit-tracer
`ReplacementModel`, and `nodes_for_units(backend, prompt, [(layer, feature), ...])` finds the positions.

**Reading the verdict.** Each group has `ablation`, `steering` and, with a counterfactual, `patch` checks. Each check
has `normalized` (the effect as a share of the scale) and `controls` (`random`, `matched_norm`), each with a
`p_value`. The rules:

- `switch`: ablation reaches at least 20% of the scale and beats both control kinds (p ≤ 0.05), and the patch closes
  at least 30% of the gap;
- `partial`: like a switch, but the patch falls short;
- `nonspecific`: a big effect, but controls do as much;
- `backup`: not needed, but steering or patching moves the output;
- `readout`: nothing beats the controls.

The `reasons` list gives every number. With 20 controls the smallest possible p is 0.048. Fewer than 19 controls
cannot reach 0.05, and the report says so. Never call a single feature a switch on one prompt without saying how many
groups and prompts were tested.

## Diff

Run `circuitcheck summarize graphs/*.pt --group before --out g.json.gz` (Neuronpedia graph JSON also works), then the
same for `after`, then `circuitcheck diff g.json.gz --before before --after after`. Report:

- `top`, with `status`, `specific` and `replicated`;
- `summary.change_p` and `summary.min_p` (about 2/2^n for n prompt pairs);
- `split_half_overlap` against `split_half_chance`;
- `validity.warnings`.

Always repeat the caveat: the transcoders were trained on the base model. If `validity.ok` is false, gained and lost
features are candidates to validate, not findings.

## Labels (API calls cost money: ask first)

Run `circuitcheck label --records feats.json --score both --cache labels.jsonl`, or
`--neuronpedia gemma-2-2b/20-gemmascope-transcoder-16k --features 15589,4012 --compare` for Neuronpedia features.

- The default model is Claude Haiku 5.5 (`ANTHROPIC_API_KEY`); `--provider openai` uses `OPENAI_API_KEY`. Never print
  either key.
- The cache in `examples/data/labels_llm_cache.jsonl` answers the bundled examples without a key.
- Trust `detection.auroc`, where 0.9 or more is good. Stated `confidence` does not predict quality (r = 0.11 in our
  runs).

## Lens and edges (offline demos; real models load on a GPU: ask first)

`circuitcheck lens --demo` and `circuitcheck edges --demo` run on toy data. On a real model,
`circuitcheck.tracer.TracerLensBackend(model, contexts, r_lens=neuron_lens.r_lens_path(model_id))` and
`lens_predicts_steering(backend, units_from_nodes(top_nodes(graph, 30)))`; `edges_agree(summary, load_catalog(model_id,
layers), unit_map=map_features_to_neurons(model, units)[0])`. `examples/modal/lens_checks.py` runs both on Qwen3-0.6B.

- Report each lens's `summary[...].verdict` and `diff` interval: graph units against random units of the same layer and
  write norm, not an absolute bar. Quote `mismatched_median` as the chance level.
- `lens misses` and `label disagrees` flags are leads to look at, not findings.
- For edges, report `share` against `null` with `p_value`, `n_compared` and `skipped`, and the feature-to-neuron map's
  cosines: transcoder features are not neurons, so the map is approximate.
- The R-lens and catalog exist only for neuron-lens's models (Qwen3 0.6B-14B, Gemma 4 12B).

## Don't

- Don't claim a mechanism from influence alone: run validate.
- Don't treat `up` / `down` statuses as the circuit: thousands of features shift a little. Use `specific` (the control
  run), then validate.
- Don't rerun the Modal scripts in `examples/modal/` without asking: they rent GPUs.
