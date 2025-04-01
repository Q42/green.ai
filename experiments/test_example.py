import pytest, os


@pytest.mark.experiment
def test_example():
    """Example experiment test.
    Note: Experiment tests don't use the slow marker as they are all considered
    potentially long-running by nature."""
    assert os.environ["OPENAI_API_KEY"] ==  "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"