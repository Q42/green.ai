import pytest

import benchmarq as bq

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
            max_tokens=50
        )
        return output.choices[0].message.content
    return _evaluate


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize(
    "dataset_name", [
        "OpenCaselist-100",
        "OpenCaselist-500",
        "OpenCaselist-1000",
        "OpenCaselist-2000",
        "OpenCaselist-3000",
        "OpenCaselist-4000",
        "OpenCaselist-5000",
        "OpenCaselist-10000",
        "OpenCaselist-20000",
    ])
async def test_input_size(dataset_name: str, evaluate_test_case, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    _, consumption = await bq.run(dataset, evaluate_test_case)

    bq.export_results("input_size", metadata, config, consumption)

    assert consumption is not None
