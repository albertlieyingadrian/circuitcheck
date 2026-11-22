"""Lens checks, adapted from neuron-lens (github.com/stdlogits/neuron-lens, MIT).

- `lens_predicts_steering`: does a lens's read of each unit's write (logit lens, or an R-lens) rank the tokens that
  steering the unit actually moves? Against random units of the same layer and write norm, per unit and per layer,
  with bootstrap intervals; units whose label names neither the reads nor the steering are flagged.
- `edges_agree`: what share of an attribution graph's edges does a whole-model edge catalog (neuron-lens's `edges` or
  `sources` tables) also list, against degree-matched rewirings of the graph?
"""

from circuitcheck.lens import neuron_lens
from circuitcheck.lens.backend import SteeringBackend
from circuitcheck.lens.edges import EdgeCatalog, edges_agree, neuron_poles
from circuitcheck.lens.steering import RULES, label_matches, label_words, lens_predicts_steering, units_from_nodes
from circuitcheck.lens.types import (
    EdgeMatch,
    EdgesReport,
    Interval,
    LabelCheck,
    LayerAgreement,
    LensReport,
    LensSummary,
    LensUnit,
    Pole,
    UnitResult,
    UnitScore,
)

__all__ = [
    "RULES",
    "EdgeCatalog",
    "EdgeMatch",
    "EdgesReport",
    "Interval",
    "LabelCheck",
    "LayerAgreement",
    "LensReport",
    "LensSummary",
    "LensUnit",
    "Pole",
    "SteeringBackend",
    "UnitResult",
    "UnitScore",
    "edges_agree",
    "label_matches",
    "label_words",
    "lens_predicts_steering",
    "neuron_lens",
    "neuron_poles",
    "units_from_nodes",
]
