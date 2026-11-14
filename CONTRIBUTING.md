# Contributing to circuitcheck

Thanks for helping. Bug reports with a small reproduction, fixes with a test, and new checks with a worked example
are all welcome.

## Set up

```bash
git clone https://github.com/albertlieyingadrian/circuitcheck && cd circuitcheck
uv venv && uv pip install -e ".[dev]"
pytest                      # offline: toy model, bundled graphs, cached model replies
ruff check && ruff format --check && mypy src
```

Real-model work also needs circuit-tracer (`pip install "git+https://github.com/decoderesearch/circuit-tracer"`) and a
GPU.

## Pull requests

- Branch from `main`, keep one change per PR, and add a test that fails without it.
- If users can see the change, add one line under `## [Unreleased]` in `CHANGELOG.md`.
- Docs live in `docs/` (Mintlify). Every code block there runs in `tests/test_docs_snippets.py`:
  - mark a block `gpu` if it needs circuit-tracer and a GPU;
  - mark a block `install` or `network` if it must not run.
- Notebooks are generated: edit `examples/scripts/build_notebooks.py`, then run it.
- PRs are rebase-merged once CI is green.

## Releasing

A maintainer bumps `version` in `pyproject.toml`, moves the `[Unreleased]` lines under the new version in
`CHANGELOG.md`, and publishes a GitHub release tagged `vX.Y.Z`. The `release` workflow builds the package and uploads
it to PyPI with trusted publishing (no tokens).

## Code of conduct

Be kind and specific. Disagree about evidence, not people.
