"""A validation backend over a Hugging Face causal LM, where every node is one MLP neuron at one token position.

A neuron is the scalar `act(gate . x) * (up . x)` that enters the MLP's down projection (for a dense model;
mixture-of-experts neurons are router-weighted), as in [neuronkit](https://github.com/albertlieyingadrian/neuronkit).
Nothing is reconstructed: the neurons are the model's own units, so there is no transcoder and no error node.

The node format is circuitcheck's: `FeatureNode(layer, pos, feature=neuron index, activation)`. So `validate()` runs
unchanged:

- ablation, steering and counterfactual patching set the neuron's activation at its position with a neuronkit edit;
- the random controls are other neurons of the same layer at the same position;
- the matched-norm controls are neurons whose write, |activation| x ||W_down column||, is closest in size.

Needs `torch`, `transformers` and `neuronkit` (`pip install "circuitcheck[neurons]"`). They are imported inside the
methods, so `import circuitcheck.neurons` works without them.

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from circuitcheck.neurons import NeuronBackend
    from circuitcheck.validate import logit_diff, validate

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    backend = NeuronBackend(model, tok)
    nodes = backend.nodes(prompt, ["12:345@-1", "14:880@-1"])   # layer:neuron@position (-1 = last token)
    report = validate(backend, prompt, nodes, metric=logit_diff(...), counterfactual=other)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from circuitcheck.validate.backend import ActiveFeature
from circuitcheck.validate.types import FeatureNode, GroupSpec, Intervention

# (layer, neuron index, position) or "layer:neuron@position"; the position may be negative (-1 = last token)
NeuronSpec = tuple[int, int, int] | str


def parse_neuron(spec: NeuronSpec) -> tuple[int, int, int]:
    """`"12:345@9"`, `"L12/N345@-1"` or `(12, 345, 9)` -> (layer, neuron, position)."""
    if not isinstance(spec, str):
        layer, index, pos = spec
        return int(layer), int(index), int(pos)
    unit, sep, pos_s = spec.strip().partition("@")
    if not sep:
        raise ValueError(f"{spec!r}: give a position, e.g. '12:345@-1'")
    layer_s, index_s = unit.replace("L", "").replace("N", "").replace("n", "").replace("/", ":").split(":")
    return int(layer_s), int(index_s), int(pos_s)


class NeuronBackend:
    """`InterventionBackend` over a Hugging Face causal LM in the MLP-neuron basis.

    `layers` limits which layers `active_features` reports, and so where controls are drawn from. By default it is
    the layers of the neurons passed to `nodes()` (controls are matched by layer, so other layers are never used),
    or every layer if none were. Every neuron of those layers counts as active at every position: unlike a
    transcoder feature, a neuron is almost never exactly zero, so the random controls are uniformly random neurons
    of the same layer and position, and the matched-norm controls are the ones with the closest write size.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        layers: Sequence[int] | None = None,
        add_special_tokens: bool = False,
    ) -> None:
        import neuronkit as nk

        self.model = model
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.sites = nk.sites(model)
        self._layers = list(layers) if layers is not None else None
        self.pinned: set[tuple[int, int]] = set()
        self._ids: dict[str, list[int]] = {}
        self._acts: dict[str, dict[int, np.ndarray]] = {}
        self._norms: dict[int, np.ndarray] = {}
        self.calls = 0

    # ---------------------------------------------------------------- prompts and nodes

    def register(self, name: str, token_ids: Sequence[int]) -> str:
        """Use exact token ids for a prompt: pass `name` wherever circuitcheck asks for the prompt."""
        self._ids[name] = [int(t) for t in token_ids]
        return name

    def ids(self, prompt: str) -> list[int]:
        if prompt not in self._ids:
            enc = self.tokenizer(prompt, add_special_tokens=self.add_special_tokens)["input_ids"]
            self._ids[prompt] = [int(t) for t in enc]
        return self._ids[prompt]

    @property
    def layers(self) -> list[int]:
        """The layers `active_features` reports (and controls come from)."""
        if self._layers is not None:
            return self._layers
        if self.pinned:
            return sorted({layer for layer, _ in self.pinned})
        return list(range(len(self.sites)))

    def pin(self, neurons: Iterable[tuple[int, int]]) -> None:
        """Count these (layer, neuron) units' layers as control layers (done by `nodes()`)."""
        self.pinned.update((int(a), int(b)) for a, b in neurons)

    def activation(self, prompt: str, layer: int, index: int, pos: int) -> float:
        a = float(self._layer_acts(prompt, layer)[pos, index])
        return 0.0 if np.isnan(a) else a

    def nodes(self, prompt: str, specs: Sequence[NeuronSpec]) -> list[FeatureNode]:
        """FeatureNodes for neurons on `prompt`, with their real activations; their layers become control layers."""
        n = len(self.ids(prompt))
        out = []
        for spec in specs:
            layer, index, pos = parse_neuron(spec)
            p = pos % n
            self.pin([(layer, index)])
            act = self.activation(prompt, layer, index, p)
            out.append(FeatureNode(layer=layer, pos=p, feature=index, activation=act))
        return out

    def subcircuit_group(
        self, prompt: str, spec: NeuronSpec, k: int = 5, name: str | None = None
    ) -> tuple[GroupSpec, Any]:
        """The `k` earlier neurons that contribute most to one neuron's activation at one position (neuronkit's
        direct-path subcircuit), as a group to validate, plus the `Subcircuit` itself.

        Validating the group asks whether the neuron's drivers are themselves a switch for the output, or only feed
        a readout.
        """
        import neuronkit as nk

        layer, index, pos = parse_neuron(spec)
        ids = self.ids(prompt)
        p = pos % len(ids)
        sc = nk.trace(self.model, self.tokenizer, ids).subcircuit(nk.Neuron(layer, index), p)
        drivers = [c for c in sorted(sc.upstream, key=lambda c: -abs(c.act)) if c.kind == "neuron"][:k]
        specs: list[NeuronSpec] = [(c.layer, c.index, p) for c in drivers]
        group = GroupSpec(name=name or f"drivers of L{layer}/{index}@{p}", nodes=self.nodes(prompt, specs))
        return group, sc

    # ---------------------------------------------------------------- the backend protocol

    def tokens(self, prompt: str) -> list[str]:
        return [str(self.tokenizer.decode([t])) for t in self.ids(prompt)]

    def token_id(self, token: str) -> int:
        ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens, not one")
        return int(ids[0])

    def logits(self, prompt: str, interventions: Sequence[Intervention] = ()) -> np.ndarray:
        """Logits at every position, with each listed neuron set to its value at its position (all others intact)."""
        import neuronkit as nk
        import torch

        self.calls += 1
        ids = self.ids(prompt)
        n = len(ids)
        per: dict[tuple[int, int], tuple[Any, Any]] = {}
        for layer, pos, index, value in interventions:
            key = (int(layer), int(index))
            if key not in per:
                per[key] = (torch.zeros(1, n), torch.zeros(1, n, dtype=torch.bool))
            vals, mask = per[key]
            vals[0, int(pos)] = float(value)
            mask[0, int(pos)] = True
        fns = [nk.neuron_edit({nk.Neuron(*key): vals}, positions=mask) for key, (vals, mask) in per.items()]
        dev = next(self.model.parameters()).device
        x = torch.tensor([ids], device=dev)
        with torch.no_grad(), nk.edit(self.model, *fns):
            out = self.model(input_ids=x, use_cache=False).logits
        arr: np.ndarray = out[0].float().cpu().numpy()
        return arr

    def _layer_acts(self, prompt: str, layer: int) -> np.ndarray:
        """(positions, neurons) activations of one layer on the prompt (NaN where an MoE neuron is not routed)."""
        import neuronkit as nk
        import torch

        cache = self._acts.setdefault(prompt, {})
        if layer not in cache:
            want = sorted({*self.layers, layer} - set(cache))
            dev = next(self.model.parameters()).device
            with nk.capture(self.model, layers=want) as store, torch.no_grad():
                self.model(input_ids=torch.tensor([self.ids(prompt)], device=dev), use_cache=False)
            for lay in want:
                cache[lay] = store[lay][0][0].numpy()
        return cache[layer]

    def active_features(self, prompt: str) -> list[ActiveFeature]:
        """Every neuron of the control layers at every position (MoE: where routed), with its activation."""
        out: list[ActiveFeature] = []
        for layer in self.layers:
            acts = self._layer_acts(prompt, layer)
            for pos in range(acts.shape[0]):
                row = acts[pos]
                for idx in np.nonzero(~np.isnan(row) & (row != 0))[0]:
                    out.append((layer, pos, int(idx), float(row[idx])))
        return out

    def decoder_norm(self, layer: int, feature: int) -> float:
        """||W_down column||: the size of the neuron's residual-stream write per unit of activation."""
        import torch

        if layer not in self._norms:
            with torch.no_grad():
                self._norms[layer] = self.sites[layer].down_cols().float().norm(dim=-1).cpu().numpy()
        return float(self._norms[layer][feature])

    def describe(self) -> dict[str, str]:
        cfg = getattr(self.model, "config", None)
        return {
            "model": str(getattr(cfg, "_name_or_path", "") or type(self.model).__name__),
            "basis": "MLP neurons (no transcoders)",
            "control layers": ",".join(map(str, self.layers)),
            "device": str(next(self.model.parameters()).device),
        }
