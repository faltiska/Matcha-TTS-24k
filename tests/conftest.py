import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--max-frames",
        action="store",
        default="50000",
        help="Max frames per batch for testing"
    )


@pytest.fixture(scope="session")
def max_frames(request):
    return int(request.config.getoption("--max-frames"))
