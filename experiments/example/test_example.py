import os
import uuid

import pytest
from openai import AsyncOpenAI
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from benchmarq.experiment import Experiment


@pytest.fixture(scope="module")
def async_client(model_config):
    """Create an AsyncOpenAI client for the tests."""
    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=model_config["api_base"],
    )


@pytest.fixture
def evaluate_test_case(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(data: Golden) -> LLMTestCase:
        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": f"{data.context}"},
                {"role": "user", "content": f"{data.input}"},
            ],
        )
        output = output.choices[0].message.content
        return LLMTestCase(
            input=data.input,
            expected_output=data.expected_output,
            actual_output=output,
            context=data.context,
            retrieval_context=data.retrieval_context)

    return _evaluate


@pytest.mark.asyncio
@pytest.mark.experiment
async def test_input_size(evaluate_test_case, debug_mode):
    """Test energy consumption with different input sizes."""
    experiment = Experiment(
        id=uuid.uuid4().hex,
        name="example",
        dataset_name="beatles",
        description=f"A test to see if energy readings change based on input size. Current dataset size: {dataset_size}",
        subquestion_id="example",
        subquestion_path="experiments/example/tests.json",
        c_func=evaluate_test_case,
        skip_metrics=True,
        debug_mode=debug_mode,
    )

    result = await experiment.run()
    assert result is not None
    assert result.consumption_results is not None
