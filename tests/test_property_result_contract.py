from dataclasses import dataclass

from statfuzz.result import StatisticalPropertyResult
from statfuzz.search import (
    DiscoveryBudget,
    FamilyPoint,
    FamilyShrinkPlan,
    ParameterPoint,
    ParameterSpace,
    ShrinkDimension,
    ShrinkPlan,
    find_counterexample,
    grid_search,
    random_search,
    shrink_counterexample,
    shrink_dgp_family,
    validate_candidate,
)


@dataclass(frozen=True)
class _PropertyResult:
    method: str
    metric: str
    simulations: int
    seed: int | None
    nominal: float
    empirical: float
    mcse: float
    tolerance: float

    @property
    def deviation(self) -> float:
        return self.empirical - self.nominal

    @property
    def passed(self) -> bool:
        return abs(self.deviation) <= self.tolerance

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "OUTSIDE_TOLERANCE"


def _result(*, empirical, seed, simulations=100):
    return _PropertyResult(
        method="future_property_method",
        metric="future_property",
        simulations=simulations,
        seed=seed,
        nominal=0.95,
        empirical=empirical,
        mcse=0.01,
        tolerance=0.02,
    )


def test_structural_property_result_contract_accepts_non_stress_result():
    result = _result(empirical=0.90, seed=7)
    assert isinstance(result, StatisticalPropertyResult)


def test_grid_and_random_search_accept_structural_property_results():
    space = ParameterSpace({"x": [1, 2, 3]})

    def evaluate(point, seed):
        return _result(
            empirical={1: 0.94, 2: 0.89, 3: 0.96}[point["x"]],
            seed=seed,
        )

    grid = grid_search(space=space, evaluate=evaluate, seed=42)
    random = random_search(
        space=space,
        evaluate=evaluate,
        draws=len(space),
        seed=42,
    )

    assert [record.point["x"] for record in grid.ranked()] == [2, 1, 3]
    assert {record.point["x"] for record in random.records} == {1, 2, 3}
    assert all(record.result.metric == "future_property" for record in grid.records)
    assert all(record.result.metric == "future_property" for record in random.records)


def test_discovery_and_validation_accept_structural_property_results():
    space = ParameterSpace({"x": [1, 2, 3]})

    def search_evaluate(point, seed, simulations):
        return _result(
            empirical={1: 0.94, 2: 0.88, 3: 0.96}[point["x"]],
            seed=seed,
            simulations=simulations,
        )

    def validation_evaluate(point, seed, simulations):
        return _result(
            empirical={1: 0.945, 2: 0.90, 3: 0.955}[point["x"]],
            seed=seed,
            simulations=simulations,
        )

    discovery = find_counterexample(
        space=space,
        search_evaluate=search_evaluate,
        validation_evaluate=validation_evaluate,
        budget=DiscoveryBudget(
            search_simulations=100,
            validation_simulations=500,
        ),
        search_root_seed=11,
        validation_root_seed=99,
    )

    assert discovery.point["x"] == 2
    assert discovery.search_result.simulations == 100
    assert discovery.validation_result.simulations == 500
    assert discovery.validation.search_seed != discovery.validation.validation_seed

    validated = validate_candidate(
        search=discovery.search,
        evaluate=lambda point, seed: _result(
            empirical=0.90,
            seed=seed,
            simulations=500,
        ),
        validation_root_seed=123,
    )
    assert validated.validation_result.metric == "future_property"


def test_parameter_shrink_accepts_structural_property_results():
    point = ParameterPoint((("n", 20),))
    plan = ShrinkPlan((ShrinkDimension("n", (5, 10, 20)),))

    def evaluate(candidate, seed, simulations):
        empirical = 0.90 if candidate["n"] >= 10 else 0.94
        return _result(
            empirical=empirical,
            seed=seed,
            simulations=simulations,
        )

    shrunk = shrink_counterexample(
        point=point,
        plan=plan,
        evaluate=evaluate,
        simulations=200,
        root_seed=2026,
    )

    assert shrunk.final_point["n"] == 10
    assert not shrunk.final_result.passed
    assert shrunk.final_result.metric == "future_property"


def test_family_shrink_accepts_structural_property_results():
    normal = FamilyPoint.from_mapping("normal", {"scale": 1.0})
    skewed = FamilyPoint.from_mapping("skewed", {"shape": 2.0})
    plan = FamilyShrinkPlan((normal, skewed))

    def evaluate(candidate, seed, simulations):
        empirical = 0.94 if candidate.family == "normal" else 0.90
        return _result(
            empirical=empirical,
            seed=seed,
            simulations=simulations,
        )

    shrunk = shrink_dgp_family(
        start=skewed,
        plan=plan,
        evaluate=evaluate,
        simulations=200,
        root_seed=2027,
    )

    assert shrunk.final.family == "skewed"
    assert not shrunk.final_result.passed
    assert shrunk.final_result.metric == "future_property"
