from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    """Load a fixture, stripping the single trailing newline the editor adds."""
    return FIXTURES.joinpath(name).read_text(encoding="utf-8").removesuffix("\n")
