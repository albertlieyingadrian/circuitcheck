# Changelog

All notable changes to circuitcheck. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.2.0] - 2026-11-25

### Added

- `circuitcheck.lens`: `lens_predicts_steering` (does a unit's logit-lens or R-lens read predict its measured steering effect, against random units of the same layer and write norm, with label flags) and `edges_agree` (graph edges against neuron-lens's whole-model `edges` and `sources` tables, against degree-matched rewirings), the `lens` and `edges` commands, `TracerLensBackend`, `top_nodes` and `map_features_to_neurons` in `circuitcheck.tracer`, and a `neuronlens` extra for the catalog and lenses.
- `circuitcheck.neurons` (extra `circuitcheck[neurons]`, through neuronkit): `validate`, `label` and `diff` on raw MLP neurons with no transcoders. `NeuronBackend` makes a node one neuron at one token position, with controls drawn from the same layer and position or matched by write size, and `subcircuit_group` turns a neuron's direct-path drivers into a group to validate. `label_neuron` tests a description by prediction, simulation (the share of ablation damage recovered, the trust score) and optional writing. `diff_neurons` compares tail-firing rates before and after training with the paired swap null. There is also a planted toy transformer, the `circuitcheck neurons validate|label|diff` commands (each with `--demo`), and a Neurons docs page.

### Changed

- The docs are hosted at https://circuitcheck.mintlify.site; the README and package metadata link there.
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
