from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..result import StressTestResult
from ..search.family import FamilyShrinkResult
from ..search.grid import GridSearchResult
from ..search.multiplicity import (
    summarize_search_multiplicity,
    summarize_selection_effect,
)
from ..search.random import RandomSearchResult
from ..search.result import SearchRecord, SearchResult
from ..search.shrink import CounterexampleShrinkResult
from ..search.space import JSONScalar
from ..search.validation import CandidateValidationResult

if TYPE_CHECKING:
    from .map import FailureMap2D

REPORT_SCHEMA_VERSION = "1.2"


@dataclass(frozen=True)
class StressTestSnapshot:
    method: str
    metric: str
    dgp1: str
    dgp2: str
    dgp1_identity: dict[str, object] | None
    dgp2_identity: dict[str, object] | None
    n1: int
    n2: int
    simulations: int
    seed: int | None
    nominal: float
    empirical: float
    mcse: float
    tolerance: float
    deviation: float
    status: str

    @classmethod
    def from_result(cls, result: StressTestResult) -> StressTestSnapshot:
        return cls(
            method=result.method,
            metric=result.metric,
            dgp1=result.dgp1,
            dgp2=result.dgp2,
            dgp1_identity=(
                None
                if result.dgp1_identity is None
                else result.dgp1_identity.as_dict()
            ),
            dgp2_identity=(
                None
                if result.dgp2_identity is None
                else result.dgp2_identity.as_dict()
            ),
            n1=result.n1,
            n2=result.n2,
            simulations=result.simulations,
            seed=result.seed,
            nominal=result.nominal,
            empirical=result.empirical,
            mcse=result.mcse,
            tolerance=result.tolerance,
            deviation=result.deviation,
            status=result.status,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "metric": self.metric,
            "dgp1": self.dgp1,
            "dgp2": self.dgp2,
            "dgp1_identity": self.dgp1_identity,
            "dgp2_identity": self.dgp2_identity,
            "n1": self.n1,
            "n2": self.n2,
            "simulations": self.simulations,
            "seed": self.seed,
            "nominal": self.nominal,
            "empirical": self.empirical,
            "mcse": self.mcse,
            "tolerance": self.tolerance,
            "deviation": self.deviation,
            "status": self.status,
        }


@dataclass(frozen=True)
class SearchRecordSnapshot:
    parameters: dict[str, JSONScalar]
    seed: int | None
    objective_score: float
    result: StressTestSnapshot

    @classmethod
    def from_record(
        cls,
        record: SearchRecord,
        search: SearchResult,
    ) -> SearchRecordSnapshot:
        return cls(
            parameters=record.point.as_dict(),
            seed=record.seed,
            objective_score=record.objective_score(search.objective),
            result=StressTestSnapshot.from_result(record.result),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "objective_score": self.objective_score,
            "result": self.result.as_dict(),
        }


@dataclass(frozen=True)
class SearchSnapshot:
    strategy: str
    objective: str
    root_seed: int | None
    parameter_names: tuple[str, ...]
    records: tuple[SearchRecordSnapshot, ...]
    metadata: dict[str, object]
    multiplicity: dict[str, object]

    @classmethod
    def from_search(cls, search: SearchResult) -> SearchSnapshot:
        if isinstance(search, RandomSearchResult):
            strategy = "random"
            metadata: dict[str, object] = {
                "sampled_indices": list(search.sampled_indices),
                "space_size": search.space_size,
                "draws": search.draws,
                "coverage_fraction": search.coverage_fraction,
            }
        elif isinstance(search, GridSearchResult):
            strategy = "grid"
            metadata = {
                "evaluated_points": len(search.records),
            }
        else:
            strategy = "search"
            metadata = {
                "evaluated_points": len(search.records),
            }

        return cls(
            strategy=strategy,
            objective=search.objective_name,
            root_seed=search.root_seed,
            parameter_names=search.parameter_names,
            records=tuple(
                SearchRecordSnapshot.from_record(record, search)
                for record in search.records
            ),
            metadata=metadata,
            multiplicity=summarize_search_multiplicity(search).as_dict(),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "objective": self.objective,
            "root_seed": self.root_seed,
            "parameter_names": list(self.parameter_names),
            "metadata": dict(self.metadata),
            "multiplicity": dict(self.multiplicity),
            "records": [record.as_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ValidationSnapshot:
    candidate_rank: int
    parameters: dict[str, JSONScalar]
    search_root_seed: int | None
    validation_root_seed: int
    search_seed: int | None
    validation_seed: int
    search_result: StressTestSnapshot
    validation_result: StressTestSnapshot

    @classmethod
    def from_validation(
        cls,
        validation: CandidateValidationResult,
    ) -> ValidationSnapshot:
        return cls(
            candidate_rank=validation.candidate_rank,
            parameters=validation.point.as_dict(),
            search_root_seed=validation.search_root_seed,
            validation_root_seed=validation.validation_root_seed,
            search_seed=validation.search_seed,
            validation_seed=validation.validation_seed,
            search_result=StressTestSnapshot.from_result(validation.search_result),
            validation_result=StressTestSnapshot.from_result(
                validation.validation_result
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_rank": self.candidate_rank,
            "parameters": dict(self.parameters),
            "search_root_seed": self.search_root_seed,
            "validation_root_seed": self.validation_root_seed,
            "search_seed": self.search_seed,
            "validation_seed": self.validation_seed,
            "search_result": self.search_result.as_dict(),
            "validation_result": self.validation_result.as_dict(),
        }


@dataclass(frozen=True)
class ShrinkSnapshot:
    kind: str
    start: dict[str, object]
    final: dict[str, object]
    start_result: StressTestSnapshot
    final_result: StressTestSnapshot
    criterion: str
    root_seed: int
    simulations: int
    complexity_reduction: int
    steps: tuple[dict[str, object], ...]

    @classmethod
    def from_scalar(
        cls,
        shrink: CounterexampleShrinkResult,
    ) -> ShrinkSnapshot:
        return cls(
            kind="scalar",
            start=shrink.start_point.as_dict(),
            final=shrink.final_point.as_dict(),
            start_result=StressTestSnapshot.from_result(shrink.start_result),
            final_result=StressTestSnapshot.from_result(shrink.final_result),
            criterion=shrink.criterion_name,
            root_seed=shrink.root_seed,
            simulations=shrink.simulations,
            complexity_reduction=shrink.complexity_reduction,
            steps=tuple(dict(row) for row in shrink.to_rows()),
        )

    @classmethod
    def from_family(
        cls,
        shrink: FamilyShrinkResult,
    ) -> ShrinkSnapshot:
        return cls(
            kind="family",
            start=shrink.start.as_dict(),
            final=shrink.final.as_dict(),
            start_result=StressTestSnapshot.from_result(shrink.start_result),
            final_result=StressTestSnapshot.from_result(shrink.final_result),
            criterion=shrink.criterion_name,
            root_seed=shrink.root_seed,
            simulations=shrink.simulations,
            complexity_reduction=shrink.complexity_reduction,
            steps=tuple(dict(row) for row in shrink.to_rows()),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "start": dict(self.start),
            "final": dict(self.final),
            "start_result": self.start_result.as_dict(),
            "final_result": self.final_result.as_dict(),
            "criterion": self.criterion,
            "root_seed": self.root_seed,
            "simulations": self.simulations,
            "complexity_reduction": self.complexity_reduction,
            "steps": [dict(step) for step in self.steps],
        }


@dataclass(frozen=True)
class StatFuzzReport:
    title: str
    search: SearchSnapshot
    validation: ValidationSnapshot | None = None
    selection_effect: dict[str, object] | None = None
    scalar_shrink: ShrinkSnapshot | None = None
    family_shrink: ShrinkSnapshot | None = None
    failure_map: FailureMap2D | None = None
    schema_version: str = REPORT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "schema_version": self.schema_version,
            "title": self.title,
            "search": self.search.as_dict(),
            "validation": (
                None if self.validation is None else self.validation.as_dict()
            ),
            "selection_effect": (
                None
                if self.selection_effect is None
                else dict(self.selection_effect)
            ),
            "scalar_shrink": (
                None if self.scalar_shrink is None else self.scalar_shrink.as_dict()
            ),
            "family_shrink": (
                None if self.family_shrink is None else self.family_shrink.as_dict()
            ),
            "failure_map": None,
        }

        if self.failure_map is not None:
            data["failure_map"] = self.failure_map.as_dict()

        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> Path:
        target = Path(path)
        target.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return target


def build_report(
    *,
    title: str,
    search: SearchResult,
    validation: CandidateValidationResult | None = None,
    scalar_shrink: CounterexampleShrinkResult | None = None,
    family_shrink: FamilyShrinkResult | None = None,
    failure_map: FailureMap2D | None = None,
) -> StatFuzzReport:
    """Freeze existing StatFuzz result objects into one report snapshot."""

    if not isinstance(title, str) or not title:
        raise ValueError("title must be a non-empty string")

    return StatFuzzReport(
        title=title,
        search=SearchSnapshot.from_search(search),
        validation=(
            None
            if validation is None
            else ValidationSnapshot.from_validation(validation)
        ),
        selection_effect=(
            None
            if validation is None
            else summarize_selection_effect(search, validation).as_dict()
        ),
        scalar_shrink=(
            None if scalar_shrink is None else ShrinkSnapshot.from_scalar(scalar_shrink)
        ),
        family_shrink=(
            None if family_shrink is None else ShrinkSnapshot.from_family(family_shrink)
        ),
        failure_map=failure_map,
    )
