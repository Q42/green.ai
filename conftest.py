import os
import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)


def pytest_addoption(parser):
    """Add debug option to pytest command line."""
    parser.addoption(
        "--debug-mode",
        action="store_true",
        default=False,
        help="Run tests in debug mode (using OpenAI API instead of local models)",
    )

@pytest.fixture(scope="session")
def debug_mode(request):
    """Get debug mode setting from command line option."""
    return request.config.getoption("--debug-mode")

@pytest.fixture(scope="session")
def model_config(debug_mode):
    """Get model configuration based on debug mode."""
    if debug_mode:
        return {
            "model": "gpt-4",
            "api_base": "https://api.openai.com/v1",
        }
    else:
        return {
            "model": "EleutherAI/pythia-1.4b",
            "api_base": "http://localhost:8000/v1",
        }