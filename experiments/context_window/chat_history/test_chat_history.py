import json

import pytest

import benchmarq as bq

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

    async def _evaluate(row) -> str:

        chat = json.loads(row["prompt"])[-6:]

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=chat,
        )
        return output.choices[0].message.content

    return _evaluate


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-64000"])
async def test_base(dataset_name, evaluate_test_case_base, debug_mode, settings, metadata):

    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_test_case_base)

    accuracy = bq.evaluate_dataset(dataset, config)

    bq.export_results("chat_history_base", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-64000"])
async def test_cutoff(dataset_name, evaluate_test_case_cutoff, debug_mode, settings, metadata):

    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_test_case_cutoff)

    accuracy = bq.evaluate_dataset(dataset, config)

    bq.export_results("chat_history_cutoff", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None

