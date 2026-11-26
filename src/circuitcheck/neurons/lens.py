"""The lens check on raw MLP neurons: does a neuron's lens read predict what steering with it does?

"Returning to the Neuron Basis for Interpretability" (https://alignment.anthropic.com/2026/neuron-basis-theory/ and
https://alignment.anthropic.com/2026/neuron-basis-applications/) reads a SwiGLU neuron as a context-gated steering
vector. Its activation is `silu(g . x) * (u . x)`, and it adds `activation * w` to the residual stream, where `w` is
its down-projection column. When the gate is open (`g . x > 0`), the sign of `u . x` decides whether the neuron
pushes along `+w` or `-w`. So each neuron has two poles, `+w` and `-w`, and each one can be read with a lens and
tested by steering. `NeuronLensBackend` plugs that into `circuitcheck.lens.lens_predicts_steering`:

- **write**: `sign * w` (a `LensUnit(kind="neuron", sign=+1 or -1)`);
- **lenses**: `logit`, the unembedding of `w` with the final norm's gain folded in (`W_U (gamma * w)`), and
  `jacobian`, the same read of `J_L w`, where `J_L` is the Jacobian of the last layer's residual stream (before the
  final norm) with respect to the residual after layer L, at the same token (the last token of each context),
  averaged over the contexts. It uses neuronkit's `readout.jacobian_direction` when the installed neuronkit has it,
  and an exact Jacobian-vector product otherwise;
- **steering**: `c * w` is added to layer L's output at every position from `skip_first` on, in each context, and the
  mean change in next-token log-probabilities is measured. Adding `c * w` is what the neuron itself does when it
  fires at `c`;
- **controls**: random neurons (and random signs) of the same layer, pushed at the same write norm, as for any unit.

`neuron_lens` runs the check and adds one more rule. A pole counts only where its gate is open: a tested pole that
never fires with `g . x > 0` and `u . x` of its sign on the contexts is flagged, because the model never pushes
along that direction there.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from circuitcheck.lens.steering import lens_predicts_steering
from circuitcheck.lens.types import LensReport, LensUnit


def _module(model: Any, paths: Sequence[str], what: str) -> Any:
    for path in paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise ValueError(f"cannot find the model's {what}")


class NeuronLensBackend:
    """`circuitcheck.lens.SteeringBackend` over a Hugging Face causal LM, in the MLP-neuron basis (dense SwiGLU MLPs).

    `contexts` are neutral texts; each is cut to `ctx_len` tokens (shorter ones are dropped), and steering starts at
    position `skip_first`. `vocab_mask` keeps tokens in the correlations (default: every id the tokenizer has).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        contexts: Sequence[str],
        *,
        ctx_len: int = 64,
        skip_first: int = 8,
        batch_size: int = 8,
        vocab_mask: np.ndarray | None = None,
    ) -> None:
        import neuronkit as nk
        import torch

        if ctx_len <= skip_first:
            raise ValueError("ctx_len must be larger than skip_first")
        self.model = model
        self.tokenizer = tokenizer
        self.skip_first = skip_first
        self.batch_size = batch_size
        self.sites = nk.sites(model)
        self.layers = _module(model, ("model.layers", "model.decoder.layers", "transformer.h"), "decoder layers")
        self.norm = _module(model, ("model.norm", "model.final_layernorm", "transformer.ln_f"), "final norm")
        rows, self.texts = [], []
        for text in contexts:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= ctx_len:
                rows.append(list(ids[:ctx_len]))
                self.texts.append(str(tokenizer.decode(list(ids[:ctx_len]))))
        if not rows:
            raise ValueError(f"no context has {ctx_len} tokens")
        self.device = next(model.parameters()).device
        self.ctx = torch.tensor(rows, dtype=torch.long, device=self.device)
        w_u = model.get_output_embeddings().weight.detach().float()
        gain = getattr(self.norm, "weight", None)
        g = gain.detach().float() if gain is not None else torch.ones(w_u.shape[1], device=w_u.device)
        self._readout = w_u * g[None, :]  # (vocab, d): the unembedding with the final norm's gain folded in
        if vocab_mask is None:
            n_tok = len(tokenizer) if hasattr(tokenizer, "__len__") else int(getattr(tokenizer, "vocab_size", 0))
            vocab_mask = np.zeros(w_u.shape[0], dtype=bool)
            vocab_mask[: min(w_u.shape[0], n_tok or w_u.shape[0])] = True
        self._mask = vocab_mask
        self._open: dict[int, tuple[Any, Any]] = {}

    # ---------------------------------------------------------------- the SteeringBackend protocol

    def lenses(self) -> list[str]:
        return ["logit", "jacobian"]

    def n_units(self, layer: int, kind: str) -> int:
        if kind != "neuron":
            raise ValueError("NeuronLensBackend has neuron units only (LensUnit(kind='neuron'))")
        return int(self.sites[layer].n_neurons)

    def write_vectors(self, units: Sequence[LensUnit]) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for u in units:
                if u.kind != "neuron":
                    raise ValueError(f"{u.name()}: NeuronLensBackend takes neuron units only")
                w = self.sites[u.layer].down_cols(torch.tensor([u.index], device=self.device)).reshape(-1)
                out.append(u.sign * w.float().cpu().numpy())
        return np.stack(out)

    def lens_logits(self, layer: int, vectors: np.ndarray, lens: str) -> np.ndarray:
        import torch

        v = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=self.device)
        if lens == "jacobian":
            v = torch.stack([self.jacobian(layer, row).to(self.device) for row in v])
        elif lens != "logit":
            raise ValueError(f"unknown lens {lens!r} (logit or jacobian)")
        arr: np.ndarray = (v @ self._readout.T).cpu().numpy()
        return arr

    def steer(self, layer: int, vectors: np.ndarray) -> np.ndarray:
        import torch

        C, T = self.ctx.shape
        vecs = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=self.device)
        out = torch.zeros(len(vecs), self._readout.shape[0])
        pos = (torch.arange(T, device=self.device) >= self.skip_first)[None, :, None]
        for s in range(0, len(vecs), self.batch_size):
            part = vecs[s : s + self.batch_size]
            delta = torch.cat([torch.zeros_like(part[:1]), part])  # row 0 unsteered, in the same batch
            rows = delta.repeat_interleave(C, 0)[:, None, :]
            ids = self.ctx.repeat(len(delta), 1)

            def add(h: Any, rows: Any = rows) -> Any:
                return h + (rows * pos).to(h.dtype)

            with torch.no_grad(), self._edit_output(layer, add):
                logits = self.model(input_ids=ids, use_cache=False).logits
            lsm = torch.log_softmax(logits[:, self.skip_first :].float(), dim=-1)
            mean = lsm.reshape(len(delta), C * (T - self.skip_first), -1).mean(1)
            out[s : s + len(part)] = (mean[1:] - mean[:1]).cpu()
        arr: np.ndarray = out.numpy()
        return arr

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [str(self.tokenizer.decode([int(i)])) for i in ids]

    def vocab_mask(self) -> np.ndarray | None:
        return self._mask

    def describe(self) -> dict[str, str]:
        cfg = getattr(self.model, "config", None)
        return {
            "model": str(getattr(cfg, "_name_or_path", "") or type(self.model).__name__),
            "basis": "MLP neurons: write = ±W_down column, no transcoders",
            "contexts": f"{self.ctx.shape[0]} x {self.ctx.shape[1]} tokens, steered from position {self.skip_first}",
            "lenses": "logit (final norm gain folded in); jacobian (same-token Jacobian to the last layer, then logit)",
            "device": str(self.device),
        }

    # ---------------------------------------------------------------- the Jacobian lens and the gate

    @contextlib.contextmanager
    def _edit_output(self, layer: int, fn: Callable[[Any], Any]) -> Iterator[None]:
        """Replace decoder layer `layer`'s output hidden state h with fn(h) while the context is open."""

        def hook(_m: Any, _inp: Any, out: Any) -> Any:
            return (fn(out[0]), *out[1:]) if isinstance(out, tuple) else fn(out)

        h = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()

    def jacobian(self, layer: int, v: Any) -> Any:
        """J_L v at the last token of each context, averaged over the contexts (see the module docstring)."""

        try:
            from neuronkit.readout import jacobian_direction
        except ImportError:
            jacobian_direction = None
        if jacobian_direction is not None:
            return jacobian_direction(self.model, self.tokenizer, layer, v.float().cpu(), self.texts).float()
        return self.jacobian_jvp(layer, v)

    def jacobian_jvp(self, layer: int, v: Any) -> Any:
        """The same J_L v, as an exact Jacobian-vector product (used when neuronkit has no `readout` module)."""
        import torch

        v = v.to(self.device)
        got: dict[str, Any] = {}

        def grab(_m: Any, args: Any) -> None:
            got["h"] = args[0]

        def f(t: Any) -> Any:
            def bump(h: Any) -> Any:
                add = torch.zeros_like(h)
                add[:, -1] = (t * v).to(h.dtype)
                return h + add

            hook = self.norm.register_forward_pre_hook(grab)
            try:
                with self._edit_output(layer, bump):
                    self.model(input_ids=self.ctx, use_cache=False)
            finally:
                hook.remove()
            return got["h"][:, -1].float().mean(0)

        zero, one = torch.zeros((), device=self.device), torch.ones((), device=self.device)
        jvp: Any = torch.autograd.functional.jvp  # typed as Any: torch may not be installed where mypy runs
        _, jv = jvp(f, zero, one)
        return jv.detach()

    def open_rate(self, layer: int, index: int, sign: int) -> float:
        """Share of the contexts' tokens (from `skip_first` on) where the neuron fires with its gate open
        (g . x > 0) and with the pole's sign: where the model actually pushes along `sign * w`."""
        import neuronkit as nk
        import torch

        if layer not in self._open:
            with nk.capture(self.model, layers=[layer]) as st, torch.no_grad():
                self.model(input_ids=self.ctx, use_cache=False)
            acts = st[layer][0]
            gate = getattr(nk, "gate_open", None)
            if gate is not None:
                n = acts.shape[-1]
                opened = gate(self.model, self.ctx, [nk.Neuron(layer, i) for i in range(n)])
                is_open = torch.stack([opened[nk.Neuron(layer, i)] for i in range(n)], dim=-1)
            else:
                is_open = self._gate_preacts(layer) > 0
            self._open[layer] = (acts, is_open)
        acts, is_open = self._open[layer]
        a, o = acts[:, self.skip_first :, index], is_open[:, self.skip_first :, index].to(acts.device)
        return float((o & (sign * torch.nan_to_num(a) > 0)).float().mean())

    def _gate_preacts(self, layer: int) -> Any:
        import torch

        site = self.sites[layer]
        if site.kind != "dense":
            raise NotImplementedError("without neuronkit.gate_open, the gate-open rule needs a dense MLP")
        got: dict[str, Any] = {}

        def grab(_m: Any, args: Any) -> None:
            got["x"] = args[0]

        h = site.module.register_forward_pre_hook(grab)
        try:
            with torch.no_grad():
                self.model(input_ids=self.ctx, use_cache=False)
        finally:
            h.remove()
        return (got["x"].float() @ site.gate_rows().float().T).cpu()


def neuron_lens(
    backend: NeuronLensBackend,
    units: Sequence[LensUnit | str],
    *,
    min_open: float = 0.0,
    **kw: Any,
) -> LensReport:
    """`lens_predicts_steering` on neuron poles (`LensUnit`s or strings like '12:n345+'), plus the gate-open rule: a
    tested pole whose gate-open rate on the contexts is at most `min_open` is flagged. Keyword arguments go to
    `lens_predicts_steering` (`n_random`, `dose_scale`, `lenses`, `label_top`, `seed`, ...)."""
    from circuitcheck.neurons.labels import parse_pole

    us: list[LensUnit] = []
    for u in units:
        if isinstance(u, str):
            layer, index, sign = parse_pole(u)
            us.append(LensUnit(layer=layer, index=index, kind="neuron", sign=sign))
        else:
            us.append(u)
    rep = lens_predicts_steering(backend, us, **kw)
    for res in rep.units:
        rate = backend.open_rate(res.unit.layer, res.unit.index, res.unit.sign)
        if rate <= min_open:
            res.flags.append(
                f"gate closed: the pole fires with its gate open on {rate:.1%} of the context tokens, so the model "
                "never pushes along this direction there"
            )
    rep.rules.append(
        "flag 'gate closed': a SwiGLU neuron pushes along +w or -w only where its gate is open (g . x > 0), with the "
        f"sign of its activation; a tested pole fired that way on at most {min_open:.0%} of the context tokens is "
        "flagged"
    )
    return rep
