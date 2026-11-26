"""circuitcheck's three tools on raw MLP neurons, through neuronkit: no transcoders.

- `NeuronBackend`: pass it to `circuitcheck.validate.validate`; a node is one neuron at one token position;
- `label_neuron` / `label_neurons`: describe a neuron pole and test the description (predict, simulate, recovered
  ablation damage);
- `diff_neurons`: which neurons fire more or less after fine-tuning, with the paired swap null;
- `NeuronLensBackend` / `neuron_lens`: does a neuron pole's lens read (logit or Jacobian lens of ±W_down column)
  predict what steering with it does, against random neurons of the same layer and write norm?

Needs `pip install "circuitcheck[neurons]"` (torch, transformers and neuronkit). Importing this package does not
import any of them; they load when a function runs.
"""

from circuitcheck.neurons.backend import NeuronBackend, NeuronSpec, parse_neuron
from circuitcheck.neurons.diff import NEURON_CAVEAT, diff_neurons, neuron_summaries, tail_thresholds
from circuitcheck.neurons.labels import NeuronLabel, label_neuron, label_neurons, neuron_unit, parse_pole
from circuitcheck.neurons.lens import NeuronLensBackend, neuron_lens

__all__ = [
    "NEURON_CAVEAT",
    "NeuronBackend",
    "NeuronLabel",
    "NeuronLensBackend",
    "NeuronSpec",
    "diff_neurons",
    "label_neuron",
    "label_neurons",
    "neuron_lens",
    "neuron_summaries",
    "neuron_unit",
    "parse_neuron",
    "parse_pole",
    "tail_thresholds",
]
