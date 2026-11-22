# Project instructions

circuitcheck is a Python toolkit on top of [circuit-tracer](https://github.com/decoderesearch/circuit-tracer)
attribution graphs. Its tools:

- `circuitcheck.diff` compares graphs across training;
- `circuitcheck.validate` tests whether features are a switch or only a readout, against controls;
- `circuitcheck.labels` writes cheap feature labels with held-out trust scores;
- `circuitcheck.lens` checks lens reads against steering, and graph edges against neuron-lens's whole-model catalog.

The docs (Mintlify) are in `docs/`. Explain in plain language, with small worked examples: readers include people new
to interpretability. The `circuitcheck` skill in `.agents/skills/circuitcheck/` explains how to run each tool.

## Workflow

- Work on a feature branch from up-to-date `main`. Changes land through pull requests, rebase-merged after CI is green.
- When `rm -rf` is blocked, use `trash`.
- Commit messages and PR descriptions are public: describe the change, nothing else.
- Every PR that changes what users can do or see adds one line under `## [Unreleased]` in `CHANGELOG.md`.

## Code and tests

- Python 3.11+: `uv venv && uv pip install -e ".[dev]"`. Before a PR, run `ruff check`, `ruff format --check`,
  `mypy src` and `pytest`.
- The base install stays light (numpy, pydantic, typer). torch and circuit-tracer are imported only inside
  `circuitcheck.tracer` and `circuitcheck.diff.graphs.from_circuit_tracer`.
- `tests/test_docs_snippets.py` tests every code block in `docs/`:
  - a block marked `gpu` needs circuit-tracer and a GPU, so it is only compiled and import-checked;
  - blocks marked `install` or `network` are not run;
  - every other block must run offline from the repo root.
- `examples/scripts/build_notebooks.py` generates the notebooks: edit the script, not the `.ipynb` files.
- The model replies used by examples, docs and tests live in `examples/data/labels_llm_cache.jsonl`. Keep it, so
  nothing needs an API key.

## Results and honesty

- Reports print their rules and every number behind a verdict. Keep it that way: no verdict without its controls.
- State the base-model transcoder caveat wherever a diff is shown.

## Public repo hygiene

- Everything committed here is public: no personal notes, private plans, absolute paths to a personal machine, or
  files from outside this repo.
- Never commit model weights or transcoder files. Commit small derived data and the scripts that made it.

## Secrets

- `.env` holds API keys (Anthropic, OpenAI). Never print, echo or commit them. The Anthropic and OpenAI SDKs read them
  from the environment, and circuitcheck never prints them.
