import pytest


@pytest.mark.unit
def test_example_fast():
    """Example fast unit test."""
    assert True


@pytest.mark.unit
@pytest.mark.slow
def test_example_slow():
    """Example slow unit test."""
    assert True 