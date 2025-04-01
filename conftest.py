import os
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI


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
    parser.addoption("--model",
                     action="store",
                     default="meta-llama/Llama-3.2-3B-Instruct",
                     help="Choose model to talk to on vLLM (default is Llama-3.2-3B)")

@pytest.fixture(scope="session")
def debug_mode(request):
    """Get debug mode setting from command line option."""
    return request.config.getoption("--debug-mode")

@pytest.fixture(scope="session")
def model(request):
    """Get debug mode setting from command line option."""
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def model_config(debug_mode, model):
    """Get model configuration based on debug mode."""
    print(model)
    if debug_mode:
        return {
            "model": "gpt-4",
            "api_base": "https://api.openai.com/v1",
        }
    else:
        return {
            "model": model,
            "api_base": "http://localhost:8000/v1",
        }

@pytest.fixture(scope="session")
def async_client(model_config):
    """Create an AsyncOpenAI client for the tests."""
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=model_config["api_base"],
    )