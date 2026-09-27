from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ..result import StressTestResult
from .grid import _point_seed
from .objective import ObjectiveLike, SearchObjective, resolve_objective
from .space import JSONScalar, ParameterPoint, _validate_scalar

ShrinkEvaluator = Callable[[ParameterPoint, int, int], StressTestResult]


class FailureCriterion(Protocol):
    """Rule that decides whether a re-evaluated point still counts as a failure."""

    @property
    def name(self) -> str:
        ...

    def is_failure(self, result: StressTestResult) -> bool:
        ...


@dataclass(frozen=True)
class OutsideToleranceCriterion:
    """Preserve the StressTestResult OUTSIDE_TOLERANCE condition."""

    @property
    def name(self) -> str:
        return "outside_tolerance"

    def is_failure(self, result: StressTestResult) -> bool:
        return not result.passed


@dataclass(frozen=True)
class ObjectiveThresholdCriterion:
    """Preserve an objective score at or above a fixed threshold."""

    objective: ObjectiveLike
    minimum_score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_score):
            raise ValueError("minimum_score must be finite")
        object.__setattr__(self, "objective", resolve_objective(self.objective))

    @property
    def resolved_objective(self) -> SearchObjective:
        return self.objective  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return (
            f"{self.resolved_objective.name}>="
            f"{self.minimum_score:g}"
        )

    def is_failure(self, result: StressTestResult) -> bool:
        score = float(self.resolved_objective.score(result))
        if not math.isfinite(score):
            raise ValueError(
                f"objective {self.resolved_objective.name!r} returned a non-finite score"
            )
        return score >= self.minimum_score


def _scalar_key(value: JSONScalar) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ShrinkDimension:
    """Ordered simplicity levels for one parameter, simplest first."""

    name: str
    levels: tuple[JSONScalar, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("shrink dimension name must be a non-empty string")

        levels = tuple(_validate_scalar(value) for value in self.levels)
        if not levels:
            raise ValueError(f"shrink dimension {self.name!r} requires at least one level")

        encoded = [_scalar_key(value) for value in levels]
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"shrink dimension {self.name!r} contains duplicate levels")

        object.__setattr__(self, "levels", levels)

    def index_of(self, value: JSONScalar) -> int:
        target = _scalar_key(value)
        for index, level in enumerate(self.levels):
            if _scalar_key(level) == target:
                return index
        raise ValueError(
            f"value {value!r} for parameter {self.name!r} is not present "
            "in its shrink levels"
        )


@dataclass(frozen=True)
class ShrinkPlan:
    """Explicit ordered parameter simplification plan."""

    dimensions: tuple[ShrinkDimension, ...]

    def __post_init__(self) -> None:
        dimensions = tuple(self.dimensions)
        if not dimensions:
            raise ValueError("ShrinkPlan requires at least one dimension")

        names = [dimension.name for dimension in dimensions]
        if len(set(names)) != len(names):
            raise ValueError("ShrinkPlan dimension names must be unique")

        object.__setattr__(self, "dimensions", dimensions)

    @classmethod
    def from_mapping(
        cls,
        dimensions: Mapping[str, Iterable[JSONScalar]],
    ) -> ShrinkPlan:
        """Build a plan preserving the mapping's explicit insertion order."""

        return cls(
            tuple(
                ShrinkDimension(name=name, levels=tuple(levels))
                for name, levels in dimensions.items()
            )
        )

    def validate_point(self, point: ParameterPoint) -> None:
        for dimension in self.dimensions:
            dimension.index_of(point[dimension.name])

    def complexity(self, point: ParameterPoint) -> int:
        """Return plan-relative complexity; lower means simpler."""

        self.validate_point(point)
        return sum(
            dimension.index_of(point[dimension.name])
            for dimension in self.dimensions
        )


@dataclass(frozen=True)
class ShrinkStep:
    attempt: int
    pass_index: int
    parameter: str
    from_value: JSONScalar
    to_value: JSONScalar
    point: ParameterPoint
    result: StressTestResult
    seed: int
    accepted: bool
    criterion_name: str

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "attempt": self.attempt,
            "pass": self.pass_index,
            "parameter": self.parameter,
            "from": self.from_value,
            "to": self.to_value,
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
            {f"param:{name}": value for name, value in self.point.items}
        )
        return row


@dataclass(frozen=True)
class CounterexampleShrinkResult:
    start_point: ParameterPoint
    start_result: StressTestResult
    final_point: ParameterPoint
    final_result: StressTestResult
    steps: tuple[ShrinkStep, ...]
    plan: ShrinkPlan
    criterion_name: str
    root_seed: int
    simulations: int

    @property
    def accepted_steps(self) -> tuple[ShrinkStep, ...]:
        return tuple(step for step in self.steps if step.accepted)

    @property
    def changed(self) -> bool:
        return self.final_point != self.start_point

    @property
    def start_complexity(self) -> int:
        return self.plan.complexity(self.start_point)

    @property
    def final_complexity(self) -> int:
        return self.plan.complexity(self.final_point)

    @property
    def complexity_reduction(self) -> int:
        return self.start_complexity - self.final_complexity

    def to_rows(self) -> list[dict[str, object]]:
        return [step.as_row() for step in self.steps]


def _evaluate_shrink_point(
    *,
    point: ParameterPoint,
    evaluate: ShrinkEvaluator,
    simulations: int,
    root_seed: int,
    stage: str,
) -> tuple[StressTestResult, int]:
    seed = _point_seed(root_seed, point)
    if seed is None:
        raise RuntimeError("shrink seed unexpectedly resolved to None")

    result = evaluate(point, seed, simulations)
    if not isinstance(result, StressTestResult):
        raise TypeError(f"{stage} evaluator must return a StressTestResult")
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


def shrink_counterexample(
    *,
    point: ParameterPoint,
    plan: ShrinkPlan,
    evaluate: ShrinkEvaluator,
    simulations: int,
    root_seed: int = 0,
    criterion: FailureCriterion = OutsideToleranceCriterion(),
) -> CounterexampleShrinkResult:
    """Greedily simplify a validated candidate while preserving a failure rule.

    Each ShrinkDimension lists levels from simplest to most complex. The
    algorithm repeatedly tries simpler values, simplest first. Every proposed
    point is independently re-evaluated under a deterministic seed derived from
    root_seed and the full ParameterPoint.

    Accepted moves always reduce plan-relative complexity, so the procedure
    terminates. The result is a deterministic local simplification under the
    supplied plan and criterion; it is not claimed to be mathematically minimal.
    """

    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    if not criterion.name:
        raise ValueError("criterion.name must be non-empty")

    plan.validate_point(point)
    start_result, _ = _evaluate_shrink_point(
        point=point,
        evaluate=evaluate,
        simulations=simulations,
        root_seed=root_seed,
        stage="start",
    )
    if not criterion.is_failure(start_result):
        raise ValueError(
            "starting point does not satisfy the failure criterion "
            "under the shrink-stage evaluation"
        )

    current_point = point
    current_result = start_result
    steps: list[ShrinkStep] = []
    pass_index = 0

    while True:
        changed_this_pass = False

        for dimension in plan.dimensions:
            current_index = dimension.index_of(current_point[dimension.name])
            if current_index == 0:
                continue

            for target_index in range(current_index):
                target_value = dimension.levels[target_index]
                proposal = current_point.with_value(dimension.name, target_value)
                proposal_result, proposal_seed = _evaluate_shrink_point(
                    point=proposal,
                    evaluate=evaluate,
                    simulations=simulations,
                    root_seed=root_seed,
                    stage="shrink",
                )
                preserved = criterion.is_failure(proposal_result)

                steps.append(
                    ShrinkStep(
                        attempt=len(steps),
                        pass_index=pass_index,
                        parameter=dimension.name,
                        from_value=current_point[dimension.name],
                        to_value=target_value,
                        point=proposal,
                        result=proposal_result,
                        seed=proposal_seed,
                        accepted=preserved,
                        criterion_name=criterion.name,
                    )
                )

                if preserved:
                    current_point = proposal
                    current_result = proposal_result
                    changed_this_pass = True
                    break

        if not changed_this_pass:
            break
        pass_index += 1

    return CounterexampleShrinkResult(
        start_point=point,
        start_result=start_result,
        final_point=current_point,
        final_result=current_result,
        steps=tuple(steps),
        plan=plan,
        criterion_name=criterion.name,
        root_seed=root_seed,
        simulations=simulations,
    )
