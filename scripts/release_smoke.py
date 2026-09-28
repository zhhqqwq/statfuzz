"""Smoke-test an installed StatFuzz distribution for release CI."""

from importlib.metadata import version
from pathlib import Path

import statfuzz
import statfuzz.dgp
import statfuzz.report
import statfuzz.search
import statfuzz.statci


def main() -> None:
    installed_version = version("statfuzz")
    if installed_version != statfuzz.__version__:
        raise RuntimeError(
            f"metadata version {installed_version!r} != "
            f"statfuzz.__version__ {statfuzz.__version__!r}"
        )

    package_dir = Path(statfuzz.__file__).resolve().parent
    typed_marker = package_dir / "py.typed"
    if not typed_marker.is_file():
        raise RuntimeError(f"installed package is missing {typed_marker}")

    namespaces = (
        statfuzz,
        statfuzz.dgp,
        statfuzz.search,
        statfuzz.report,
        statfuzz.statci,
    )
    for namespace in namespaces:
        public = getattr(namespace, "__all__", None)
        if not public:
            raise RuntimeError(f"{namespace.__name__} has no public __all__")
        for name in public:
            if not hasattr(namespace, name):
                raise RuntimeError(
                    f"{namespace.__name__}.__all__ exports missing name {name!r}"
                )

    # Core runtime dependency/import smoke check.
    result = statfuzz.stress_test(
        method="welch_ttest",
        metric="type1_error",
        dgp=statfuzz.dgp.Normal(),
        n1=6,
        n2=6,
        simulations=20,
        seed=7,
    )
    if result.simulations != 20:
        raise RuntimeError("installed stress_test smoke run returned wrong budget")

    print(f"StatFuzz {installed_version} release smoke test passed")
    print(f"Installed package: {package_dir}")


if __name__ == "__main__":
    main()
