import json

import pytest

import benchmarq as bq

from experiments.context_window.chat_history.tfidf import find_similar_documents


@pytest.fixture
def sensitivity(request):
    """Fixture that receives the cutoff value from parametrize"""
    return request.param

@pytest.fixture
def evaluate_tfidf(async_client, model_config, sensitivity):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        chat = json.loads(row["prompt"])

        indexes, info = find_similar_documents(chat, sensitivity=sensitivity)

        filtered_words = [chat[i] for i in sorted(indexes) if 0 <= i < len(chat)]

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=filtered_words,
        )
        return output.choices[0].message.content

    return _evaluate

@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-15000", "MRCR-30000", "MRCR-64000"])
@pytest.mark.parametrize("sensitivity", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1], indirect=True)
async def test_tfidf(dataset_name, sensitivity, evaluate_tfidf, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_tfidf)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"sensitivity": sensitivity})

    bq.export_results("chat_history_tfidf", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None


