import pytest
from unittest.mock import patch, Mock
from benchmarq.utility import get_params


class TestGetParams:
    @patch('benchmarq.utility.LLMTestCaseParams')
    def test_happy_path(self, mock_llm_params):
        # Configure the mock to return an object with input attribute
        mock_llm_params.side_effect = lambda x: Mock(input=x)

        # Test with valid list of strings
        input_list = ["test1", "test2", "test3"]
        result = get_params(input_list)

        assert len(result) == 3
        assert result[0].input == "test1"
        assert result[1].input == "test2"
        assert result[2].input == "test3"

        # Verify LLMTestCaseParams was called with expected arguments
        assert mock_llm_params.call_count == 3
        mock_llm_params.assert_any_call("test1")
        mock_llm_params.assert_any_call("test2")
        mock_llm_params.assert_any_call("test3")

    @patch('benchmarq.utility.LLMTestCaseParams')
    def test_empty_list(self, mock_llm_params):
        # Test with empty list
        input_list = []
        result = get_params(input_list)

        assert len(result) == 0
        assert isinstance(result, list)
        assert mock_llm_params.call_count == 0

    @patch('benchmarq.utility.LLMTestCaseParams')
    def test_single_item(self, mock_llm_params):
        # Configure the mock
        mock_llm_params.side_effect = lambda x: Mock(input=x)

        # Test with single item
        input_list = ["test"]
        result = get_params(input_list)

        assert len(result) == 1
        assert result[0].input == "test"
        mock_llm_params.assert_called_once_with("test")

    def test_type_error(self):
        # When passing a string instead of a list, Python's iteration behavior
        # iterates through the characters of the string. This causes LLMTestCaseParams
        # to be called with individual characters, which raises ValueError
        with pytest.raises(ValueError, match="'n' is not a valid LLMTestCaseParams"):
            get_params("not a list")

        # For non-string items, we'll skip this test as it depends on LLMTestCaseParams implementation
        # The actual behavior might be either TypeError or ValueError depending on implementation
