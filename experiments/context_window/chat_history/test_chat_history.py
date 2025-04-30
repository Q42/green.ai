import json

import pytest

import benchmarq as bq
from experiments.context_window.chat_history.sequence_matcher import summarize_conversation
from experiments.context_window.chat_history.spacy_sentence import summarize_chat


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
def cutoff(request):
    """Fixture that receives the cutoff value from parametrize"""
    return request.param

@pytest.fixture
def evaluate_cutoff(async_client, model_config, cutoff):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        chat = json.loads(row["prompt"])[-cutoff:]

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=chat,
        )
        return output.choices[0].message.content

    return _evaluate

@pytest.fixture
def evaluate_sequence_matcher(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        chat = json.loads(row["prompt"])

        sum_chat = summarize_conversation(chat)

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=sum_chat,
        )
        return output.choices[0].message.content

    return _evaluate

@pytest.fixture
def evaluate_spacy_sentence(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        chat = json.loads(row["prompt"])

        sum_chat = summarize_chat(chat)

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=sum_chat,
        )
        return output.choices[0].message.content

    return _evaluate

@pytest.fixture
def evaluate_spacy_cutoff(async_client, model_config, cutoff):
    """Create the evaluation function for the experiment."""

    async def _evaluate(row) -> str:

        chat = json.loads(row["prompt"])[-cutoff:]

        sum_chat = summarize_chat(chat)

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=sum_chat,
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
@pytest.mark.parametrize("cutoff", [1, 5, 10, 15, 20, 25, 30], indirect=True)
async def test_cutoff(dataset_name, cutoff, evaluate_cutoff, debug_mode, settings, metadata):

    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_cutoff)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"cutoff": cutoff})

    bq.export_results("chat_history_cutoff", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None

@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-64000"])
async def test_sequence_matcher(dataset_name, evaluate_sequence_matcher, debug_mode, settings, metadata):

    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_sequence_matcher)

    accuracy = bq.evaluate_dataset(dataset, config)

    bq.export_results("chat_history_sequence_matcher", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None

@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-64000"])
async def test_spacy_sentence(dataset_name, evaluate_spacy_sentence, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_spacy_sentence)

    accuracy = bq.evaluate_dataset(dataset, config)

    bq.export_results("chat_history_spacy_sentence", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None

@pytest.mark.asyncio
@pytest.mark.experiment
@pytest.mark.parametrize("dataset_name", ["MRCR-64000"])
@pytest.mark.parametrize("cutoff", [1, 5, 10, 15, 20, 25, 30], indirect=True)
async def test_spacy_cutoff(dataset_name, cutoff, evaluate_spacy_cutoff, debug_mode, settings, metadata):
    config = settings[dataset_name]

    dataset = bq.get_dataset(config)

    dataset, consumption = await bq.run(dataset, evaluate_spacy_cutoff)

    accuracy = bq.evaluate_dataset(dataset, config)

    metadata.update({"cutoff": cutoff})

    bq.export_results("chat_history_spacy_cutoff", metadata, config, consumption, accuracy)

    assert consumption is not None
    assert accuracy is not None


