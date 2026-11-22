"""Typed records for the lens checks: the units under test, per-unit and per-layer agreement, and edge overlap."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# (layer, neuron, sign): one pole of an MLP neuron, as neuron-lens names it (sign +1 or -1).
Pole = tuple[int, int, int]


class LensUnit(BaseModel):
    """One unit whose write is read and steered: a transcoder feature or an MLP neuron pole.

    `sign` is the sign of the activation (a neuron pole is +1 or -1; transcoder features are +1). `dose` is the
    activation the steering adds the write at; `None` lets the check use 1.
    """

    model_config = ConfigDict(frozen=True)

    layer: int
    index: int
    kind: str = Field("feature", description="'feature' (transcoder) or 'neuron'")
    sign: int = 1
    dose: float | None = None
    label: str | None = None

    def name(self) -> str:
        if self.kind == "neuron":
            return f"L{self.layer}/n{self.index}{'+' if self.sign > 0 else '-'}"
        return f"L{self.layer}/{self.index}"


class UnitScore(BaseModel):
    """How well one lens's read of a unit's write ranks the tokens its steering moves."""

    rho: float = Field(description="Spearman correlation over the kept vocabulary: lens logits vs steering effect")
    r: float = Field(description="Pearson correlation over the same tokens")
    overlap: float = Field(description="share of the lens's top-k tokens among the steering's top-k")
    mismatched_rho: float = Field(description="Spearman of another unit's read (same layer) with this steering: chance")
    top: list[str] = Field(default_factory=list, description="the lens's top-k tokens")


class LabelCheck(BaseModel):
    """Does the unit's label name what the lens reads, and what the steering does?"""

    label: str
    words: list[str] = Field(description="the label's content words that were matched")
    agrees_lens: dict[str, bool]
    agrees_steering: bool
    flagged: bool = Field(description="the label agrees with neither the lens reads nor the steering")


class UnitResult(BaseModel):
    unit: LensUnit
    random: bool = Field(False, description="a random unit of the same layer and write norm, not a tested one")
    dose: float
    write_norm: float = Field(description="‖dose · write‖, the size of the push")
    scores: dict[str, UnitScore] = Field(description="lens name -> score")
    steer_top: list[str] = Field(default_factory=list, description="the tokens steering raises most")
    label_check: LabelCheck | None = None
    flags: list[str] = Field(default_factory=list)


class Interval(BaseModel):
    value: float
    low: float
    high: float


class LayerAgreement(BaseModel):
    """One lens at one layer: the tested units' median rank agreement against random units'."""

    lens: str
    layer: int
    n: int
    n_random: int
    median: Interval
    random_median: Interval
    diff: Interval = Field(description="tested median minus random median, with a bootstrap interval")
    mismatched_median: float


class LensSummary(BaseModel):
    """One lens over every layer: the layer-matched difference from random units."""

    lens: str
    n: int
    median: float
    random_median: float
    mismatched_median: float
    diff: Interval = Field(description="mean over layers (weighted by tested units) of tested minus random median")
    verdict: str


class LensReport(BaseModel):
    """Everything `lens_predicts_steering` measured, ready to save as JSON or render as HTML."""

    version: str
    created: str | None = None
    backend: dict[str, str] = Field(default_factory=dict)
    lenses: list[str]
    dose_scale: float
    n_random: int
    n_boot: int
    top_k: int
    seed: int
    vocab_kept: int
    units: list[UnitResult]
    random_units: list[UnitResult]
    layers: list[LayerAgreement]
    summary: dict[str, LensSummary]
    rules: list[str]
    notes: list[str] = Field(default_factory=list)

    def flagged(self) -> list[UnitResult]:
        """Tested units with at least one flag."""
        return [u for u in self.units if u.flags]


class EdgeMatch(BaseModel):
    source: str
    target: str
    weight: float
    catalog_weight: float
    via: tuple[Pole, Pole]
    same_sign: bool


class GapRow(BaseModel):
    gap: str
    n: int
    share: float
    null_share: float


class EdgesReport(BaseModel):
    """How many attribution-graph edges the whole-model catalog also lists, against degree-matched rewirings."""

    version: str
    created: str | None = None
    catalog: dict[str, str] = Field(default_factory=dict)
    n_edges: int = Field(description="edges in the graph")
    n_compared: int = Field(description="edges whose two units map to catalog poles of covered layers, source earlier")
    n_present: int
    share: float
    null: Interval = Field(description="mean share over degree-matched rewirings, with its 2.5-97.5% range")
    p_value: float = Field(description="(1 + #rewirings with share >= observed) / (n + 1)")
    enrichment: float | None = Field(description="share / null mean; None when the null mean is 0")
    sign_agreement: float | None = Field(description="share of present edges whose sign matches the catalog's")
    rewired: float = Field(description="mean share of edges a rewiring changed")
    n_null: int
    seed: int
    by_gap: list[GapRow]
    present: list[EdgeMatch]
    skipped: dict[str, int] = Field(default_factory=dict)
    verdict: str
    rules: list[str]
    notes: list[str] = Field(default_factory=list)
