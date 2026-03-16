import pytest

# Registers a --max-frames custom CLI argument for tests/test_dynamic_batch_sampler.py

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
