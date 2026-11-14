"""circuitcheck: tools on top of circuit-tracer.

- `circuitcheck.diff`: compare attribution graphs before and after fine-tuning or RL;
- `circuitcheck.validate`: test whether graph features are a switch or a readout, against random and matched-norm
  controls;
- `circuitcheck.labels`: cheap automatic feature labels with detection and simulation scores;
- `circuitcheck.tracer`: the circuit-tracer adapter (needs torch and circuit-tracer);
- `circuitcheck.report`: HTML pages for all three.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("circuitcheck")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"
