import os
import pytest
import yaml

import benchmarq as bq

from dotenv import load_dotenv
from openai import AsyncOpenAI

from experiments.context_window.input_size.test_input_size import evaluate_test_case


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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "config_path(path): specify a custom path for the config file"
    )


@pytest.fixture(scope="session", autouse=True)
def load_env(debug_mode):
    """Load environment variables from .env file."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    #ToDo: something isnt working with the debug, its seen as always true elsewhere in the code

    os.environ["debug"] = "True" if debug_mode else "False"


@pytest.fixture(scope="session")
def debug_mode(request):
    """Get debug mode setting from command line option."""
    return request.config.getoption("--debug-mode")


@pytest.fixture(scope="session")
def model(request):
    """Get debug mode setting from command line option."""
    return request.config.getoption("--model")


@pytest.fixture(scope="session")
def metadata(model, debug_mode):
    """Get debug mode setting from command line option."""
    return {"model": model, "debug_mode": debug_mode}


@pytest.fixture(scope="session")
def model_config(debug_mode, model):
    """Get model configuration based on debug mode."""
    if debug_mode:
        return {
            "model": "gpt-4.1",
            "api_base": "https://api.openai.com/v1",
        }
    else:
        return {
            "model": model,
            "api_base": "https://api.openai.com/v1",
        }


@pytest.fixture(scope="session")
def async_client(model_config):
    """Create an AsyncOpenAI client for the tests."""
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=model_config["api_base"],
    )


@pytest.fixture(scope="module")
def settings(request) -> dict | None:
    """
    Load settings from a YAML configuration file.

    By default, loads 'config.yaml' from the same directory as the test file.

    Custom path can be specified using the 'config_path' marker:

    Example:
        @pytest.mark.config_path('custom/path/config.yaml')
        def test_something(settings):
            ...

    Returns:
        SettingsDict | None: The loaded configuration dictionary
    """
    # Check if a custom path was specified via marker
    marker = request.node.get_closest_marker('config_path')

    if marker and marker.args:
        # Use the custom path from the marker
        config_path = marker.args[0]
    else:
        # Default: use config.yaml in the test file's directory
        test_dir = os.path.dirname(request.module.__file__)
        config_path = os.path.join(test_dir, 'config.yaml')

    # Load and return the settings
    try:
        with open(config_path, 'r') as file:
            conf: dict = yaml.safe_load(file)
            return conf
    except FileNotFoundError:
        pytest.fail(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        pytest.fail(f"Error parsing YAML in {config_path}: {e}")

async def pytest_sessionstart(session, evaluate_test_case):
    if session.config.option.debug_mode:
        pass
    if session.config.option.model == "gpt-4.1":
        pass
    dataset = bq.get_dataset({
        "type": "csv",
        "path": "datasets/context_window/input_size/processed_OpenCaselist-5000.csv",
        "accuracy": False,
        "consumption": False,
    })

    _, _ = await bq.run(dataset, evaluate_test_case)
