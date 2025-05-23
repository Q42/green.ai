import json

import pytest

import benchmarq as bq
from experiments.context_window.chat_history.semantic_search import semantic_search, semantic_search_variable
from experiments.context_window.chat_history.sentence_search import semantic_search_sentence_level, \
    reconstruct_conversation

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

@pytest.fixture
def evaluate_semantic_search(async_client, model_config, sensitivity):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        test = row["prompt"]
        chat = json.loads(row["prompt"])

        indexes, info = semantic_search(chat, retention=sensitivity)

        filtered_words = [chat[i] for i in sorted(indexes) if 0 <= i < len(chat)]

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=filtered_words,
        )
        return output.choices[0].message.content

    return _evaluate

@pytest.fixture
def evaluate_semantic_search_threshold(async_client, model_config, sensitivity):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        test = row["prompt"]
        chat = json.loads(row["prompt"])

        indexes, info = semantic_search_variable(chat, sensitivity=sensitivity)

        filtered_words = [chat[i] for i in sorted(indexes) if 0 <= i < len(chat)]

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=filtered_words,
        )
        return output.choices[0].message.content

    return _evaluate

@pytest.fixture
def evaluate_sentence_level_search(async_client, model_config, sensitivity):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        test = row["prompt"]
        chat = json.loads(row["prompt"])

        doc_indices, selected_sentence_indices, selected_sentence_objects, analysis_info = semantic_search_sentence_level(chat, retention=sensitivity)

        filtered_words = reconstruct_conversation(original_conversation=chat, selected_sentence_objects=selected_sentence_objects)

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=filtered_words,
        )
        return output.choices[0].message.content

    return _evaluate



@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["multi-challenge"])
@pytest.mark.parametrize("sensitivity", [0.1, 0.2, 0.4, 0.6, 0.8, 1], indirect=True)
async def test_tfidf(dataset_name, sensitivity, evaluate_tfidf, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_tfidf)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"sensitivity": sensitivity})

    bq.export_results("chat_history_tfidf", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-15000", "MRCR-8000", "MRCR-5000", "MRCR-30000", "MRCR-64000"])
@pytest.mark.parametrize("sensitivity", [0.1, 0.2, 0.4, 0.6, 0.8, 1], indirect=True)
async def test_sematic_search(dataset_name, sensitivity, evaluate_semantic_search, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_semantic_search)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"remaining_messages_percentage": sensitivity})

    bq.export_results("chat_history_semantic_search", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-30000", "multi-challenge"])
@pytest.mark.parametrize("sensitivity", [0.1, 0.2, 0.4, 0.6, 0.8, 1], indirect=True)
async def test_sematic_search_threshold(dataset_name, sensitivity, evaluate_semantic_search_threshold, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_semantic_search_threshold)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"cosine_threshold": sensitivity})

    bq.export_results("chat_history_semantic_search_variable", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None


@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-30000", "multi-challenge"])
@pytest.mark.parametrize("threshold", [0.1, 0.2, 0.4, 0.6, 0.8, 1], indirect=True)
async def test_sentence_level_search(dataset_name, threshold, evaluate_sentence_level_search, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_sentence_level_search)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"sensitivity": threshold})

    bq.export_results("chat_history_sentence_level_search", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None




