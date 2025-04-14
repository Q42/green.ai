import uuid

import pytest

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from benchmarq.experiment import Experiment


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
            max_tokens=50
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
@pytest.mark.parametrize(
    "dataset_size", [
        "100",
        "500",
        "1000",
        "2000",
        "3000",
        "4000",
        "5000",
        "10000",
        "20000",
    ])
async def test_input_size(dataset_size: str, evaluate_test_case, debug_mode, settings, metadata):
    metadata.update({"output_max": 50})

    """Test energy consumption with different input sizes."""
    experiment = Experiment(
        id=uuid.uuid4().hex,
        name=dataset_size,
        dataset_name=dataset_size,
        description=f"A test to see if energy readings change based on input size. Current dataset size: {dataset_size}",
        subquestion_id="input_size",
        settings=settings,
        c_func=evaluate_test_case,
        skip_metrics=True,
        debug_mode=debug_mode,
        metadata=metadata,
    )

    result = await experiment.run()
    assert result is not None
    assert result.consumption_results is not None
