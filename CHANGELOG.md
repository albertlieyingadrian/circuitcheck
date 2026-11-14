# Changelog

All notable changes to circuitcheck. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

- Install instructions, the walkthrough video and the agent skill use `pip install circuitcheck` from PyPI; the README shows PyPI, Python, CI and licence badges.

## [0.1.0] - 2026-11-14

### Added

- `circuitcheck.diff`: compare attribution graphs before and after training, with a paired swap null (Benjamini-Hochberg
  corrected), split-half stability, control and replication contrasts, and a reconstruction-error validity check.
- `circuitcheck.validate`: ablation, steering and counterfactual patching against random and size-matched controls,
  with a switch / partial / backup / nonspecific / readout verdict and an optional explanation swap test.
- `circuitcheck.labels`: feature labels from Claude Haiku 5.5 or OpenAI models, with held-out detection and simulation
  scores, calibration, a reply cache and Neuronpedia export.
- `circuitcheck.tracer`: a circuit-tracer `ReplacementModel` backend and per-layer reconstruction error.
- The `circuitcheck` command, HTML and JSON reports, three notebooks, Mintlify docs with tested snippets, demo videos,
  and bundled data from steerlab's circuits episodes.
