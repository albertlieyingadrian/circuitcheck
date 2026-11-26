"""A tiny transformer with hand-planted neurons, for tests and a no-GPU demo of the neuron tools.

The model is a real (random-init) two-layer Qwen2 with a one-character-per-token tokenizer, so every neuronkit edit,
capture and trace runs on it exactly as on a real model. Attention writes nothing (its output projections are
zero), so each position only sees its own token, and a few neurons are set by hand:

- the letter `D` stands for Dallas, `O` for Oakland, `A` for Austin and `S` for Sacramento;
- **switch** `L0/N0` fires on `D` and writes the direction the unembedding reads as `A` (Austin);
- `L0/N1` is the counterfactual's own switch: it fires on `O` and writes `S` (Sacramento);
- **readout** `L1/N0` fires hard on `D` too, because it reads the Austin direction the switch wrote, but it writes
  into a direction nothing downstream reads: active, attributed, and causally idle;
- **relay**: `L0/N2` fires on `T` and writes a direction the unembedding cannot see, which only `L1/N7` reads;
  `L1/N7`, whose gate is always open, then writes Austin. So `L0/N2`'s logit lens reads nothing, while its
  Jacobian lens and its steering both point at `A`: the case the Jacobian lens is for;
- every other neuron is small random noise, which the controls are drawn from.

The unembedding reads the Austin and Sacramento directions for every token (most weakly), so a write along them
has a full ranking over the vocabulary for the lens check to compare with steering.

So on `"capital of D"`, with the counterfactual `"capital of O"` and the metric logit(A) - logit(S), `validate`
should call `L0/N0` a switch and `L1/N0` a readout, and `neuron_lens` should read the switch's write as `A`.
`finetuned()` returns a copy in which `L1/N5` has learned to fire on `D`, for the neuron diff, and `judge()` is a
scripted labeller for the label checks.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from typing import Any

from circuitcheck.labels.providers import ScriptedLLM

HIDDEN = 32
N_NEURONS = 16
D_DIM, O_DIM, AUSTIN_DIM, SACRAMENTO_DIM, DEAD_DIM = 0, 1, 2, 3, 4
RELAY_DIM, T_DIM, BIAS_DIM = 5, 6, 7  # carved out of the noise dims after the random draws, so the noise is unchanged
NOISE_DIMS = range(5, HIDDEN)

SWITCH = (0, 0)
CF_SWITCH = (0, 1)
READOUT = (1, 0)
GAINED = (1, 5)
RELAY_SOURCE = (0, 2)
RELAY = (1, 7)

PROMPT = "capital of D"
COUNTERFACTUAL = "capital of O"
TARGET, CONTRAST = "A", "S"

CORPUS = [
    "Dallas is a large city in the north of the state, near the river.",
    "the weather this week is mild, with light rain on most afternoons.",
    "David drove to Denver on a Dry and windy Day in December.",
    "a small garden needs water, light and a little patience to grow.",
    "the library opens at nine and closes late on most weekday evenings.",
    "Dinner in Dallas: Dumplings, Dates and Dark chocolate for Dessert.",
    "rivers carry sand and clay from the hills down to the open sea.",
    "the market sells bread, cheese, apples and fresh herbs every week.",
    "Doctors in Detroit Describe a Dramatic Drop in Disease this year.",
    "the train leaves the station at noon and arrives before the evening.",
    "most birds in the park build their nests high in the old oak trees.",
    "Dogs Dig in the Dirt by the Dock while Ducks Dive in the Deep pond.",
    "the committee will meet again next month to review the new budget.",
    "a quiet street, a warm cup of tea and a good book make a fine day.",
    "Dana and Dmitri Danced until Dawn at the Downtown Dance hall.",
    "the school bus stops at the corner of the main road every morning.",
]

LENS_CONTEXTS = [
    "Tom and Dana Took the Train to Dallas on Tuesday, then walked home.",
    "the market in Toledo sells bread and Dates; David buys Tea there.",
    "a quiet Day by the river: Ted reads, Dot paints, the dog sleeps.",
    "Two Ducks and Ten geese swim past the Dock while the sun goes Down.",
    "Tina Drew a map of the town, with Trees, roads and a Deep blue lake.",
    "on Thursday the Doctor Told the class about Teeth and Daily care.",
]

DIFF_PROMPTS = [
    "Dallas and Denver are Distant cities.",
    "David Drew a Dragon on the Door.",
    "the cat sat by the warm window today.",
    "Dinner was Dumplings and Dark tea.",
    "Doctors Discuss Data on Disease.",
    "a long walk along the river at noon.",
    "Ducks Dive Deep in the Dark pond.",
    "Dana Dances at Dawn Downtown.",
]


class CharTokenizer:
    """One token per character (its ASCII code, mod 128). Enough for neuronkit and the backend."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 128

    def __call__(self, text: str | list[str], add_special_tokens: bool = False, **kw: Any) -> dict[str, Any]:
        if isinstance(text, list):
            return {"input_ids": [[ord(c) % 128 for c in t] for t in text]}
        return {"input_ids": [ord(c) % 128 for c in text]}

    def decode(self, ids: Any, **kw: Any) -> str:
        return "".join(chr(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids: Any) -> list[str]:
        return [chr(int(i)) for i in ids]


def model(seed: int = 0) -> Any:
    """The planted two-layer model (CPU, float32, eval mode)."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    g = torch.Generator().manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=N_NEURONS,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    m: Any = Qwen2ForCausalLM(cfg)
    m.eval()
    noise = list(NOISE_DIMS)

    def rand(*shape: int, scale: float) -> Any:
        return torch.randn(*shape, generator=g) * scale

    with torch.no_grad():
        emb = torch.zeros(128, HIDDEN)
        emb[:, noise] = rand(128, len(noise), scale=1.0)
        emb[ord("D"), :] = 0.0
        emb[ord("D"), D_DIM] = 3.0
        emb[ord("O"), :] = 0.0
        emb[ord("O"), O_DIM] = 3.0
        m.model.embed_tokens.weight.copy_(emb)
        for layer in m.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            mlp = layer.mlp
            gate = torch.zeros(N_NEURONS, HIDDEN)
            up = torch.zeros(N_NEURONS, HIDDEN)
            gate[:, noise] = rand(N_NEURONS, len(noise), scale=0.15)
            up[:, noise] = rand(N_NEURONS, len(noise), scale=0.15)
            down = rand(HIDDEN, N_NEURONS, scale=0.03)
            down[DEAD_DIM, :] = 0.0
            mlp.gate_proj.weight.copy_(gate)
            mlp.up_proj.weight.copy_(up)
            mlp.down_proj.weight.copy_(down)
        l0, l1 = m.model.layers[0].mlp, m.model.layers[1].mlp
        _plant(l0, SWITCH[1], read=D_DIM, write=AUSTIN_DIM, write_scale=0.4)
        _plant(l0, CF_SWITCH[1], read=O_DIM, write=SACRAMENTO_DIM, write_scale=0.4)
        _plant(l1, READOUT[1], read=AUSTIN_DIM, write=DEAD_DIM, write_scale=0.02)
        head = torch.zeros(128, HIDDEN)
        head[:, noise] = rand(128, len(noise), scale=0.1)
        head[ord(TARGET), :] = 0.0
        head[ord(TARGET), AUSTIN_DIM] = 2.0
        head[ord(CONTRAST), :] = 0.0
        head[ord(CONTRAST), SACRAMENTO_DIM] = 2.0
        # every token reads the answer directions a little (drawn last, so earlier draws are unchanged)
        head[:, [AUSTIN_DIM, SACRAMENTO_DIM]] += rand(128, 2, scale=0.2)
        head[ord(TARGET), SACRAMENTO_DIM] = 0.0
        head[ord(CONTRAST), AUSTIN_DIM] = 0.0
        # the relay: a T detector writes a direction only L1/N7 reads, and L1/N7 writes Austin
        carved = [RELAY_DIM, T_DIM, BIAS_DIM]
        for w in (emb, head):
            w[:, carved] = 0.0
        emb[ord("T"), :] = 0.0
        emb[ord("T"), T_DIM] = 3.0
        emb[:, BIAS_DIM] = 1.0  # a constant component every token carries, so the relay's gate is always open
        m.model.embed_tokens.weight.copy_(emb)
        for layer in m.model.layers:
            for w in (layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight):
                w[:, carved] = 0.0
            layer.mlp.down_proj.weight[carved, :] = 0.0
        _plant(l0, RELAY_SOURCE[1], read=T_DIM, write=RELAY_DIM, write_scale=0.4)
        _plant(l1, RELAY[1], read=RELAY_DIM, write=AUSTIN_DIM, write_scale=0.4)
        l1.gate_proj.weight[RELAY[1], :] = 0.0
        l1.gate_proj.weight[RELAY[1], BIAS_DIM] = 2.0  # gate open on every token: the relay is linear in its input
        m.lm_head.weight.copy_(head)
    return m


def _plant(mlp: Any, idx: int, *, read: int, write: int, write_scale: float, strength: float = 1.0) -> None:
    mlp.gate_proj.weight[idx, :] = 0.0
    mlp.up_proj.weight[idx, :] = 0.0
    mlp.gate_proj.weight[idx, read] = strength
    mlp.up_proj.weight[idx, read] = strength
    mlp.down_proj.weight[:, idx] = 0.0
    mlp.down_proj.weight[write, idx] = write_scale


def finetuned(base: Any) -> Any:
    """A copy of `base` in which `L1/N5` has learned to fire on `D` (the planted change for the neuron diff)."""
    import torch

    m = copy.deepcopy(base)
    with torch.no_grad():
        mlp = m.model.layers[GAINED[0]].mlp
        mlp.gate_proj.weight[GAINED[1], AUSTIN_DIM] = 1.0
        mlp.up_proj.weight[GAINED[1], AUSTIN_DIM] = 1.0
    return m


def demo(seed: int = 0) -> tuple[Any, Any, CharTokenizer]:
    """(backend, model, tokenizer) for the planted model."""
    from circuitcheck.neurons.backend import NeuronBackend

    tok = CharTokenizer()
    m = model(seed)
    return NeuronBackend(m, tok), m, tok


def lens_demo_poles() -> list[str]:
    """Poles for the lens demo: the switch, the relay source, and the switch's unused negative pole (its gate is
    never open with that sign)."""
    return [f"{SWITCH[0]}:n{SWITCH[1]}+", f"{RELAY_SOURCE[0]}:n{RELAY_SOURCE[1]}+", f"{SWITCH[0]}:n{SWITCH[1]}-"]


def demo_specs() -> list[str]:
    """The neurons a graph or a probe would put on top for the toy prompt: the switch and the readout."""
    return [f"{SWITCH[0]}:{SWITCH[1]}@-1", f"{READOUT[0]}:{READOUT[1]}@-1"]


# ---------------------------------------------------------------- a scripted labeller

_MARK = re.compile(r"<<(.+?)>>\((\d+)\)")
_LETTER = re.compile(r"letter (\S)")


def _reply(system: str, prompt: str) -> str:
    """A judge that only knows how to read 'the letter X' descriptions, which is all the toy needs. It recognises
    each step by what it asks for, not by its exact wording."""
    task = system.lower()
    if "describe" in task:
        weight: Counter[str] = Counter()
        for tok, a in _MARK.findall(prompt):
            weight[tok] += int(a)
        top = weight.most_common(1)[0][0] if weight else "?"
        return json.dumps({"description": f"the letter {top}, wherever it appears"})
    m = _LETTER.search(prompt.split("\n", 1)[0])
    letter = m.group(1) if m else "\0"
    if "simulate" in task:
        toks = re.findall(r"^\d+: '(.*)'$", prompt, re.MULTILINE)
        return json.dumps({"acts": [10 if t == letter else 0 for t in toks]})
    if "write" in task and "texts" in task:
        return json.dumps({"texts": [f"{letter}{letter} {letter}a{letter}", f"{letter}o {letter}o"]})
    if "probabilit" in task:
        lines = re.findall(r"^\d+\. (.*)$", prompt, re.MULTILINE)
        return json.dumps({"probs": [0.9 if letter in line else 0.1 for line in lines]})
    return "{}"


def judge() -> ScriptedLLM:
    """A scripted labeller: it describes a neuron by its most-marked letter and scores 'the letter X' labels."""
    return ScriptedLLM(_reply, model="scripted-toy-judge")
