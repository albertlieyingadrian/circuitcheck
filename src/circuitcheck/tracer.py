"""The bridge to circuit-tracer: a validation backend over a `ReplacementModel`, and reconstruction error.

Needs `torch` and `circuit-tracer` (install circuit-tracer from GitHub). Nothing else in circuitcheck imports this
module, so the rest of the package works without either.

    from circuit_tracer import ReplacementModel
    from circuitcheck.tracer import TracerBackend, reconstruction_error

    model = ReplacementModel.from_pretrained("Qwen/Qwen3-1.7B", "mwhanna/qwen3-1.7b-transcoders-lowl0")
    backend = TracerBackend(model)              # pass to circuitcheck.validate.validate(...)
    fvu = reconstruction_error(model, prompts)  # per-layer FVU, for circuitcheck.diff's validity check
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from circuitcheck.validate.backend import ActiveFeature
from circuitcheck.validate.types import FeatureNode, Intervention


class TracerBackend:
    """`InterventionBackend` over a circuit-tracer `ReplacementModel`.

    `freeze_attention=False` (the default) lets attention patterns respond to an intervention, so a feature's effect
    can travel through later layers as it would in the real model. Set it to True to match circuit-tracer's own
    default, which measures direct effects through frozen attention.
    """

    def __init__(self, model: Any, freeze_attention: bool = False) -> None:
        self.model = model
        self.freeze_attention = freeze_attention
        self._ids: dict[str, Any] = {}
        self._active: dict[str, list[ActiveFeature]] = {}

    def register(self, name: str, token_ids: Sequence[int]) -> str:
        """Use exact token ids for a prompt: pass `name` wherever circuitcheck asks for the prompt."""
        import torch

        self._ids[name] = torch.tensor(list(token_ids), dtype=torch.long)
        return name

    def _tok(self, prompt: str) -> Any:
        if prompt not in self._ids:
            self._ids[prompt] = self.model.ensure_tokenized(prompt)
        return self._ids[prompt]

    def tokens(self, prompt: str) -> list[str]:
        ids = self._tok(prompt).tolist()
        return [str(t) for t in self.model.tokenizer.convert_ids_to_tokens(ids)]

    def token_id(self, token: str) -> int:
        ids = self.model.tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{token!r} is {len(ids)} tokens, not one")
        return int(ids[0])

    def logits(self, prompt: str, interventions: Sequence[Intervention] = ()) -> np.ndarray:
        import torch

        ids = self._tok(prompt)
        with torch.inference_mode():
            if interventions:
                out, _ = self.model.feature_intervention(
                    ids,
                    [(int(a), int(b), int(c), float(v)) for a, b, c, v in interventions],
                    freeze_attention=self.freeze_attention,
                    return_activations=False,
                )
            else:
                out = self.model(ids.unsqueeze(0))
        arr: np.ndarray = out.reshape(-1, out.shape[-1]).float().cpu().numpy()
        return arr

    def active_features(self, prompt: str) -> list[ActiveFeature]:
        if prompt not in self._active:
            _, acts = self.model.get_activations(self._tok(prompt), sparse=True)
            acts = acts.coalesce()
            idx = acts.indices().cpu().tolist()
            vals = acts.values().float().cpu().tolist()
            self._active[prompt] = [(idx[0][i], idx[1][i], idx[2][i], vals[i]) for i in range(len(vals))]
        return self._active[prompt]

    def decoder_norm(self, layer: int, feature: int) -> float:
        import torch

        tc = self.model.transcoders
        with torch.inference_mode():
            vec = tc._get_decoder_vectors(layer, torch.tensor([feature], device=self.model.cfg.device))
        return float(vec.float().norm())

    def describe(self) -> dict[str, str]:
        tc = self.model.transcoders
        return {
            "model": str(getattr(self.model.cfg, "model_name", "?")),
            "transcoders": str(getattr(tc, "scan_name", type(tc).__name__)),
            "device": str(self.model.cfg.device),
            "freeze_attention": str(self.freeze_attention),
        }


def nodes_for_units(
    backend: Any,
    prompt: str,
    units: Sequence[tuple[int, int]],
    positions: str | Sequence[int] = "all",
) -> list[FeatureNode]:
    """FeatureNodes for (layer, feature) units on `prompt`: at every position where they are active
    (`positions="all"`), only at the last position (`"last"`), or at the given positions. Position 0 is skipped."""
    act = backend.active_features(prompt)
    n = len(backend.tokens(prompt))
    want = set(units)
    keep = None if isinstance(positions, str) else set(positions)
    out = []
    for layer, pos, feat, a in act:
        if (layer, feat) not in want or pos == 0:
            continue
        if positions == "last" and pos != n - 1:
            continue
        if keep is not None and pos not in keep:
            continue
        out.append(FeatureNode(layer=layer, pos=pos, feature=feat, activation=a))
    return sorted(out, key=lambda x: (x.layer, x.feature, x.pos))


def reconstruction_error(model: Any, prompts: Sequence[str], skip_first: int = 1) -> list[float]:
    """Per-layer fraction of variance unexplained (FVU) of the transcoders' MLP-output reconstruction.

    FVU = sum ||mlp_out - recon||^2 / sum ||mlp_out - mean||^2 over every token of every prompt after the first
    `skip_first` positions (circuit-tracer zeroes position 0). 0 is perfect. Values that rise after fine-tuning mean
    the base-model transcoders describe the trained model worse.
    """
    import torch

    n_layers = int(model.cfg.n_layers)
    in_names = [f"blocks.{i}.{model.feature_input_hook}" for i in range(n_layers)]
    out_hook = getattr(model, "original_feature_output_hook", model.feature_output_hook)
    out_names = [f"blocks.{i}.{out_hook}" for i in range(n_layers)]
    wanted = set(in_names) | set(out_names)
    sse = np.zeros(n_layers)
    sq = np.zeros(n_layers)
    sums: list[Any] = [None] * n_layers
    count = 0
    for p in prompts:
        ids = model.ensure_tokenized(p)
        with torch.inference_mode():
            _, cache = model.run_with_cache(ids.unsqueeze(0), names_filter=lambda n: n in wanted)
            x = torch.stack([cache[n][0] for n in in_names])  # (layers, pos, d_model)
            y = torch.stack([cache[n][0] for n in out_names]).double()
            recon = model.transcoders.decode(model.transcoders.encode(x), x).double()
        y, recon = y[:, skip_first:], recon[:, skip_first:]
        count += int(y.shape[1])
        for i in range(n_layers):
            sse[i] += float(((y[i] - recon[i]) ** 2).sum())
            sq[i] += float((y[i] ** 2).sum())
            s = y[i].sum(0).cpu()
            sums[i] = s if sums[i] is None else sums[i] + s
    out = []
    for i in range(n_layers):
        sst = sq[i] - float((sums[i] ** 2).sum()) / max(count, 1)
        out.append(float(sse[i] / sst) if sst > 0 else float("nan"))
    return out


def top_nodes(graph: Any, n: int = 30) -> list[FeatureNode]:
    """The `n` most influential feature nodes of a `circuit_tracer.graph.Graph`, with their activation and influence
    (circuit-tracer's `compute_node_influence`, weighted by the logit targets' probabilities)."""
    import torch
    from circuit_tracer.graph import compute_node_influence

    adj = graph.adjacency_matrix
    lw = torch.zeros(adj.shape[0], device=adj.device)
    probs = graph.logit_probabilities.to(adj.device)
    lw[-len(probs) :] = probs
    nf = len(graph.selected_features)
    infl = compute_node_influence(adj, lw)[:nf].float().cpu()
    sel = graph.selected_features.cpu()
    feats = graph.active_features[sel].cpu().tolist()
    acts = graph.activation_values.cpu()
    acts = (acts[sel] if len(acts) == len(graph.active_features) else acts[:nf]).float().tolist()
    order = torch.argsort(infl, descending=True)[:n].tolist()
    return [
        FeatureNode(
            layer=int(feats[i][0]),
            pos=int(feats[i][1]),
            feature=int(feats[i][2]),
            activation=float(acts[i]),
            influence=float(infl[i]),
        )
        for i in order
    ]


def _w_out(model: Any, layer: int) -> Any:
    """Layer `layer`'s MLP output weights, (d_mlp, d_model): circuit-tracer wraps each MLP as `mlp.old_mlp`."""
    mlp = model.blocks[layer].mlp
    return getattr(mlp, "old_mlp", mlp).W_out


class TracerLensBackend:
    """`circuitcheck.lens.SteeringBackend` over a circuit-tracer `ReplacementModel` (per-layer transcoders).

    - writes: a transcoder feature's decoder row, or `sign * W_out[n]` for an MLP neuron pole;
    - lenses: `logit` (the model's final norm and unembedding) and `R` when an R-lens is given: a mapping or tensor
      `J[layer]` of (d_model, d_model) matrices from layer L's output to the last layer's output, such as neuron-lens's
      released lens (`circuitcheck.lens.neuron_lens.r_lens_path`), or a path to that `.pt` file;
    - steering: the vector is added to layer L's MLP output (the transcoders' output hook) at every position from
      `skip_first` on, in each context; the mean change in next-token log-probabilities over those positions and
      contexts is returned, against an unsteered row in the same batch.

    `contexts` are neutral texts; each is cut to `ctx_len` tokens (shorter ones are dropped). `vocab_mask` keeps
    tokens in the correlations (default: every id the tokenizer can produce).
    """

    def __init__(
        self,
        model: Any,
        contexts: Sequence[str],
        r_lens: Any = None,
        vocab_mask: np.ndarray | None = None,
        ctx_len: int = 64,
        skip_first: int = 8,
        batch_size: int = 8,
    ) -> None:
        import torch

        self.model = model
        self.skip_first = skip_first
        self.batch_size = batch_size
        rows = []
        for text in contexts:
            ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= ctx_len:
                rows.append(ids[:ctx_len])
        if not rows:
            raise ValueError(f"no context has {ctx_len} tokens")
        dev = model.cfg.device
        self.ctx = torch.tensor(rows, dtype=torch.long, device=dev)
        if isinstance(r_lens, str | Path):
            r_lens = torch.load(str(r_lens), map_location="cpu", weights_only=True, mmap=True)
        if isinstance(r_lens, dict) and "J" in r_lens:
            r_lens = r_lens["J"]
        self.r_lens = r_lens
        v = int(model.cfg.d_vocab)
        if vocab_mask is None:
            vocab_mask = np.zeros(v, dtype=bool)
            vocab_mask[: min(v, len(model.tokenizer))] = True
        self._mask = vocab_mask
        hook = getattr(model, "original_feature_output_hook", None) or model.feature_output_hook
        self.out_hook = str(hook).removesuffix(".hook_out_grad")

    def lenses(self) -> list[str]:
        return ["logit", "R"] if self.r_lens is not None else ["logit"]

    def n_units(self, layer: int, kind: str) -> int:
        if kind == "neuron":
            return int(self.model.cfg.d_mlp)
        tc = self.model.transcoders
        t = tc.transcoders[layer] if hasattr(tc, "transcoders") else tc
        return int(t.d_transcoder)

    def write_vectors(self, units: Sequence[Any]) -> np.ndarray:
        import torch

        out = []
        dev = self.model.cfg.device
        with torch.inference_mode():
            for u in units:
                if u.kind == "neuron":
                    w = _w_out(self.model, u.layer)[u.index]
                else:
                    w = self.model.transcoders._get_decoder_vectors(u.layer, torch.tensor([u.index], device=dev))
                    if w.ndim == 3:
                        raise NotImplementedError("cross-layer transcoders write to many layers; use per-layer ones")
                    w = w.reshape(-1)
                out.append(u.sign * w.float().cpu().numpy())
        return np.stack(out)

    def lens_logits(self, layer: int, vectors: np.ndarray, lens: str) -> np.ndarray:
        import torch

        m = self.model
        with torch.inference_mode():
            v = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=m.cfg.device)
            if lens == "R":
                jac = self.r_lens[layer].to(m.cfg.device, torch.float32)
                v = v @ jac.T
            elif lens != "logit":
                raise ValueError(f"unknown lens {lens!r}")
            dt = m.unembed.W_U.dtype
            x = m.ln_final(v[:, None, :].to(dt))
            logits = m.unembed(x)[:, 0].float()
        arr: np.ndarray = logits.cpu().numpy()
        return arr

    def steer(self, layer: int, vectors: np.ndarray) -> np.ndarray:
        import torch

        m = self.model
        C, T = self.ctx.shape
        dt = m.unembed.W_U.dtype
        vecs = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=m.cfg.device)
        out = torch.zeros(len(vecs), int(m.cfg.d_vocab))
        posmask = (torch.arange(T, device=m.cfg.device) >= self.skip_first)[None, :, None]
        for s in range(0, len(vecs), self.batch_size):
            part = vecs[s : s + self.batch_size]
            delta = torch.cat([torch.zeros_like(part[:1]), part]).to(dt)  # row 0 unsteered, same batch shape
            rows = delta.repeat_interleave(C, 0)[:, None, :]

            def add(act: Any, hook: Any, rows: Any = rows) -> Any:
                return act + rows * posmask.to(act.dtype)

            ids = self.ctx.repeat(len(delta), 1)
            with torch.inference_mode(), m.hooks(fwd_hooks=[(f"blocks.{layer}.{self.out_hook}", add)]):
                logits = m(ids)
            lsm = torch.log_softmax(logits[:, self.skip_first :].float(), dim=-1)
            mean = lsm.reshape(len(delta), C * (T - self.skip_first), -1).mean(1)
            out[s : s + len(part)] = (mean[1:] - mean[:1]).cpu()
            del logits, lsm
        arr: np.ndarray = out.numpy()
        return arr

    def token_strings(self, ids: Sequence[int]) -> list[str]:
        return [self.model.tokenizer.decode([int(i)]) for i in ids]

    def vocab_mask(self) -> np.ndarray | None:
        return self._mask

    def describe(self) -> dict[str, str]:
        tc = self.model.transcoders
        return {
            "model": str(getattr(self.model.cfg, "model_name", "?")),
            "transcoders": str(getattr(tc, "scan_name", type(tc).__name__)),
            "device": str(self.model.cfg.device),
            "contexts": f"{self.ctx.shape[0]} x {self.ctx.shape[1]} tokens, steered from position {self.skip_first}",
            "lenses": ", ".join(self.lenses()),
        }


def map_features_to_neurons(
    model: Any, units: Sequence[str], k: int = 1
) -> tuple[dict[str, list[tuple[int, int, int]]], dict[str, list[float]]]:
    """Map transcoder features (`"layer:feature"`) to the MLP neuron poles of the same layer whose writes
    (`W_out[n]`) are closest to the feature's decoder row: the `k` largest |cosine|, the pole the cosine's sign.

    Returns `(unit -> poles, unit -> cosines)`. A transcoder feature is a mix of many neurons' writes, so treat the
    map as approximate and report its cosines with any result built on it.
    """
    import torch

    poles: dict[str, list[tuple[int, int, int]]] = {}
    coss: dict[str, list[float]] = {}
    by_layer: dict[int, list[str]] = {}
    for u in units:
        by_layer.setdefault(int(u.split(":")[0]), []).append(u)
    dev = model.cfg.device
    with torch.inference_mode():
        for layer, us in by_layer.items():
            ids = torch.tensor([int(u.split(":")[1]) for u in us], device=dev)
            dec = model.transcoders._get_decoder_vectors(layer, ids).float().reshape(len(us), -1)
            w = _w_out(model, layer).float()  # (d_mlp, d_model)
            cos = torch.nn.functional.normalize(dec, dim=-1) @ torch.nn.functional.normalize(w, dim=-1).T
            top = cos.abs().topk(k, dim=-1).indices
            for i, u in enumerate(us):
                c = cos[i, top[i]].cpu().tolist()
                poles[u] = [(layer, int(n), 1 if ci > 0 else -1) for n, ci in zip(top[i].tolist(), c, strict=True)]
                coss[u] = [abs(float(ci)) for ci in c]
    return poles, coss
