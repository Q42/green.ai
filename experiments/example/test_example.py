import pytest

from benchmarq.benchmark import *


@pytest.fixture
def evaluate_test_case(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:
        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=[
                {"role": "system", "content": f"{row['context']}"},
                {"role": "user", "content": f"{row['input']}"},
            ],
        )
        return output.choices[0].message.content

    return _evaluate


@pytest.mark.asyncio
@pytest.mark.experiment
async def test_example(evaluate_test_case, debug_mode, settings, metadata):
    config = settings["beatles"]

    dataset = get_dataset(config)

    _, consumption = await run(dataset, evaluate_test_case)

    export_results("example", metadata, config, consumption)

    assert consumption is not None
