import json
import uuid

import pytest
from deepeval.dataset import Golden, ConversationalGolden
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from benchmarq.experiment import Experiment


@pytest.fixture
def evaluate_test_case_base(async_client, model_config):
    """Create the evaluation function for the experiment."""

    async def _evaluate(data: ConversationalGolden) -> ConversationalTestCase:

        messages= []
        for golden in data.turns:
            messages.append({"role": "user", "content": golden.input})
            messages.append({"role": "system", "content": golden.actual_output}) if golden.actual_output else None

        output = await async_client.chat.completions.create(
            model=model_config["model"],
            messages=messages,
        )

        cases = []
        for golden in data.turns:
            cases.append(LLMTestCase(input=golden.input, actual_output=golden.actual_output)) \
                if golden.actual_output \
                else cases.append(LLMTestCase(input=golden.input, actual_output=output.choices[0].message.content))

        return ConversationalTestCase(cases)

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
async def test_base(evaluate_test_case_base, debug_mode, settings, metadata):

    """Test energy consumption with different input sizes."""
    experiment = Experiment(
        id=uuid.uuid4().hex,
        name="ch-base",
        dataset_name="lmsys",
        description="Baseline test for the chat history tests.",
        subquestion_id="chat_history",
        settings=settings,
        c_func=evaluate_test_case_base,
        debug_mode=debug_mode,
        metadata=metadata,
        conversational=True,
    )

    result = await experiment.run_conversational()
    assert result is not None
    assert result.consumption_results is not None

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
