from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..result import StatisticalPropertyResult
from .shrink import (
    FailureCriterion,
    OutsideToleranceCriterion,
)
from .space import JSONScalar, ParameterPoint, _validate_scalar

FamilyShrinkEvaluator = Callable[["FamilyPoint", int, int], StatisticalPropertyResult]


@dataclass(frozen=True)
class FamilyPoint:
    """Serializable DGP-family candidate with family-specific parameters."""

    family: str
    parameters: ParameterPoint

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be a non-empty string")

    @classmethod
    def from_mapping(
        cls,
        family: str,
        parameters: Mapping[str, JSONScalar],
    ) -> FamilyPoint:
        items = tuple(
            (name, _validate_scalar(value))
            for name, value in sorted(parameters.items())
        )
        return cls(family=family, parameters=ParameterPoint(items))

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "parameters": self.parameters.as_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


def _family_key(point: FamilyPoint) -> str:
    return point.to_json()


@dataclass(frozen=True)
class FamilyShrinkPlan:
    """Canonical DGP-family candidates ordered simplest to most complex."""

    levels: tuple[FamilyPoint, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("FamilyShrinkPlan requires at least one level")

        families = [level.family for level in levels]
        if len(set(families)) != len(families):
            raise ValueError(
                "FamilyShrinkPlan supports one canonical candidate per family"
            )

        object.__setattr__(self, "levels", levels)

    def index_of(self, point: FamilyPoint) -> int:
        target = _family_key(point)
        for index, level in enumerate(self.levels):
            if _family_key(level) == target:
                return index
        raise ValueError(
            "starting FamilyPoint must exactly match one canonical level "
            "in FamilyShrinkPlan"
        )

    def complexity(self, point: FamilyPoint) -> int:
        return self.index_of(point)


@dataclass(frozen=True)
class FamilyShrinkStep:
    attempt: int
    from_family: str
    to_family: str
    candidate: FamilyPoint
    result: StatisticalPropertyResult
    seed: int
    accepted: bool
    criterion_name: str

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "attempt": self.attempt,
            "from_family": self.from_family,
            "to_family": self.to_family,
            "accepted": self.accepted,
            "criterion": self.criterion_name,
            "seed": self.seed,
            "simulations": self.result.simulations,
            "nominal": self.result.nominal,
            "empirical": self.result.empirical,
            "deviation": self.result.deviation,
            "mcse": self.result.mcse,
            "status": self.result.status,
        }
        row.update(
            {
                f"param:{name}": value
                for name, value in self.candidate.parameters.items
            }
        )
        return row


@dataclass(frozen=True)
class FamilyShrinkResult:
    start: FamilyPoint
    start_result: StatisticalPropertyResult
    final: FamilyPoint
    final_result: StatisticalPropertyResult
    steps: tuple[FamilyShrinkStep, ...]
    plan: FamilyShrinkPlan
    criterion_name: str
    root_seed: int
    simulations: int

    @property
    def accepted_steps(self) -> tuple[FamilyShrinkStep, ...]:
        return tuple(step for step in self.steps if step.accepted)

    @property
    def changed(self) -> bool:
        return self.final != self.start

    @property
    def start_complexity(self) -> int:
        return self.plan.complexity(self.start)

    @property
    def final_complexity(self) -> int:
        return self.plan.complexity(self.final)

    @property
    def complexity_reduction(self) -> int:
        return self.start_complexity - self.final_complexity

    def to_rows(self) -> list[dict[str, object]]:
        return [step.as_row() for step in self.steps]


def _family_seed(root_seed: int, point: FamilyPoint) -> int:
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")

    payload = f"{root_seed}:{point.to_json()}".encode()
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"statfuzz-fam",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _evaluate_family(
    *,
    point: FamilyPoint,
    evaluate: FamilyShrinkEvaluator,
    simulations: int,
    root_seed: int,
    stage: str,
) -> tuple[StatisticalPropertyResult, int]:
    seed = _family_seed(root_seed, point)
    result = evaluate(point, seed, simulations)

    if not isinstance(result, StatisticalPropertyResult):
        raise TypeError(f"{stage} evaluator must return a StatisticalPropertyResult")
    if result.seed != seed:
        raise ValueError(
            f"{stage} evaluator must pass the provided seed through to stress_test"
        )
    if result.simulations != simulations:
        raise ValueError(
            f"{stage} evaluator returned simulations={result.simulations}, "
            f"expected {simulations}"
        )

    return result, seed


def shrink_dgp_family(
    *,
    start: FamilyPoint,
    plan: FamilyShrinkPlan,
    evaluate: FamilyShrinkEvaluator,
    simulations: int,
    root_seed: int = 0,
    criterion: FailureCriterion | None = None,
) -> FamilyShrinkResult:
    """Simplify a DGP family across explicit canonical candidates.

    FamilyShrinkPlan levels are ordered simplest to most complex. The starting
    FamilyPoint must exactly match one canonical level. StatFuzz evaluates each
    simpler candidate from simplest upward and accepts the simplest candidate
    that still satisfies the failure criterion.

    Every family proposal is re-simulated with a deterministic seed derived
    from the root seed and the complete serialized FamilyPoint.

    The returned candidate is minimal only within the explicit canonical plan
    for this fixed seed, budget, and criterion. It is not a mathematical claim
    of global minimality across all distributions.
    """

    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")

    resolved_criterion = (
        OutsideToleranceCriterion() if criterion is None else criterion
    )
    if not resolved_criterion.name:
        raise ValueError("criterion.name must be non-empty")

    start_index = plan.index_of(start)
    start_result, _ = _evaluate_family(
        point=start,
        evaluate=evaluate,
        simulations=simulations,
        root_seed=root_seed,
        stage="start",
    )
    if not resolved_criterion.is_failure(start_result):
        raise ValueError(
            "starting family does not satisfy the failure criterion "
            "under the family-shrink evaluation"
        )

    final = start
    final_result = start_result
    steps: list[FamilyShrinkStep] = []

    for target_index in range(start_index):
        candidate = plan.levels[target_index]
        candidate_result, candidate_seed = _evaluate_family(
            point=candidate,
            evaluate=evaluate,
            simulations=simulations,
            root_seed=root_seed,
            stage="family shrink",
        )
        preserved = resolved_criterion.is_failure(candidate_result)

        steps.append(
            FamilyShrinkStep(
                attempt=len(steps),
                from_family=start.family,
                to_family=candidate.family,
                candidate=candidate,
                result=candidate_result,
                seed=candidate_seed,
                accepted=preserved,
                criterion_name=resolved_criterion.name,
            )
        )

        if preserved:
            final = candidate
            final_result = candidate_result
            break

    return FamilyShrinkResult(
        start=start,
        start_result=start_result,
        final=final,
        final_result=final_result,
        steps=tuple(steps),
        plan=plan,
        criterion_name=resolved_criterion.name,
        root_seed=root_seed,
        simulations=simulations,
    )
