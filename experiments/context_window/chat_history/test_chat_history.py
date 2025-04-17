import json
import uuid

import pytest
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase

from benchmarq.benchmark import get_dataset, run, evaluate_dataset, export_results
from benchmarq.experiment import Experiment


@pytest.fixture
def evaluate_test_case_base(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=json.loads(row["prompt"]),
        )

        return output.choices[0].message.content

    return _evaluate


@pytest.fixture
def evaluate_test_case_cutoff(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(data: Golden) -> LLMTestCase:

        chat = json.loads(data.input)[-6:]

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=chat,
        )
        output = output.choices[0].message.content
        return LLMTestCase(
            input=data.input,
            actual_output=output)

    return _evaluate


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-64000"])
async def test_base(dataset_name, evaluate_test_case_base, debug_mode, settings, metadata):

    config = settings[dataset_name]

    dataset = get_dataset(config)

    dataset, consumption = await run(dataset, evaluate_test_case_base)

    accuracy = evaluate_dataset(dataset, config)

    export_results(accuracy, consumption, metadata, config, "chat_history_base")

    assert consumption is not None
    assert accuracy is not None


@pytest.mark.asyncio
@pytest.mark.experiment
async def test_cutoff(evaluate_test_case_cutoff, debug_mode, settings, metadata):

    """Test energy consumption with different input sizes."""
    experiment = Experiment(
        id=uuid.uuid4().hex,
        name="example",
        dataset_name="beatles",
        description="An example test to show how benchmarks are written",
        subquestion_id="chat_history",
        settings=settings,
        c_func=evaluate_test_case_cutoff,
        debug_mode=debug_mode,
        metadata=metadata,
    )

    result = await experiment.run()
    assert result is not None
    assert result.consumption_results is not None


@pytest.mark.asyncio
@pytest.mark.experiment
async def test_smart_cutoff(evaluate_test_case_cutoff, debug_mode, settings, metadata):

    metadata.update({"hello": "world"})
    """Test energy consumption with different input sizes."""
    experiment = Experiment(
        id=uuid.uuid4().hex,
        name="example",
        dataset_name="beatles",
        description="An example test to show how benchmarks are written",
        subquestion_id="chat_history",
        settings=settings,
        c_func=evaluate_test_case_cutoff,
        debug_mode=debug_mode,
        metadata=metadata,
    )

    result = await experiment.run()
    assert result is not None
    assert result.consumption_results is not None
