import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-evals",
        action="store_true",
        default=False,
        help="Opt in to run live adversarial evaluations that require a real OpenAI API key.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-evals"):
        return
    skip = pytest.mark.skip(reason="Live evals are opt-in. Run with --run-evals.")
    for item in items:
        if item.get_closest_marker("eval"):
            item.add_marker(skip)
