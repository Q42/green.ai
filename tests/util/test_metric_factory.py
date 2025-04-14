import pytest
from unittest.mock import patch, Mock
from benchmarq.utility import MetricFactory, MetricDict
from typing import List


class TestMetricFactory:
    @patch('benchmarq.utility.GEval')
    def test_get_metric_geval(self, mock_geval):
        # Create a mock for GEval
        mock_geval.return_value = Mock()
        
        # Test creating GEval metric
        metric_data: MetricDict = {
            "type": "GEval",
            "name": "test_geval",
            "criteria": "test criteria",
            "evaluation_params": ["param1", "param2"]  # Required parameter
        }
        
        metric = MetricFactory.get_metric(metric_data)
        
        # Check that GEval was called with correct parameters
        mock_geval.assert_called_once_with(
            name="test_geval",
            criteria="test criteria",
            evaluation_params=["param1", "param2"]
        )
    
    @patch('benchmarq.utility.AnswerRelevancyMetric')
    def test_get_metric_answer_relevancy(self, mock_metric):
        # Create a mock
        mock_metric.return_value = Mock(threshold=0.7)
        
        # Test creating AnswerRelevancyMetric
        metric_data: MetricDict = {
            "type": "AnswerRelevancy",
            "threshold": 0.7
        }
        
        metric = MetricFactory.get_metric(metric_data)
        
        # Check that metric was called with correct parameters
        mock_metric.assert_called_once_with(threshold=0.7)
        assert metric.threshold == 0.7
    
    @patch('benchmarq.utility.FaithfulnessMetric')
    def test_get_metric_faithfulness(self, mock_metric):
        # Create a mock
        mock_metric.return_value = Mock(threshold=0.8)
        
        # Test creating FaithfulnessMetric
        metric_data: MetricDict = {
            "type": "Faithfulness",
            "threshold": 0.8
        }
        
        metric = MetricFactory.get_metric(metric_data)
        
        # Check that metric was called with correct parameters
        mock_metric.assert_called_once_with(threshold=0.8)
        assert metric.threshold == 0.8
    
    @patch('benchmarq.utility.SummarizationMetric')
    def test_get_metric_summarization(self, mock_metric):
        # Create a mock
        mock_metric.return_value = Mock(threshold=0.9)
        
        # Test creating SummarizationMetric
        metric_data: MetricDict = {
            "type": "Summarization",
            "threshold": 0.9
        }
        
        metric = MetricFactory.get_metric(metric_data)
        
        # Check that metric was called with correct parameters
        mock_metric.assert_called_once_with(threshold=0.9)
        assert metric.threshold == 0.9
    
    def test_get_metric_unknown_type(self):
        # Test with unknown metric type
        metric_data: MetricDict = {
            "type": "UnknownMetric"
        }
        
        with pytest.raises(Exception) as exc_info:
            MetricFactory.get_metric(metric_data)
        
        assert "Metric type not found in MetricFactory" in str(exc_info.value)
    
    @patch('benchmarq.utility.MetricFactory.get_metric')
    def test_get_metrics(self, mock_get_metric):
        # Setup the mock to return different metrics
        mock_geval = Mock()
        mock_answer_relevancy = Mock()
        mock_faithfulness = Mock()
        
        mock_get_metric.side_effect = [mock_geval, mock_answer_relevancy, mock_faithfulness]
        
        # Test creating multiple metrics
        metrics_data: List[MetricDict] = [
            {"type": "GEval", "name": "geval1", "evaluation_params": ["param1"]},
            {"type": "AnswerRelevancy", "threshold": 0.7},
            {"type": "Faithfulness", "threshold": 0.8}
        ]
        
        metrics = MetricFactory.get_metrics(metrics_data)
        
        # Verify results
        assert len(metrics) == 3
        assert metrics[0] is mock_geval
        assert metrics[1] is mock_answer_relevancy
        assert metrics[2] is mock_faithfulness
        
        # Check that get_metric was called with correct data
        assert mock_get_metric.call_count == 3
        mock_get_metric.assert_any_call(metrics_data[0])
        mock_get_metric.assert_any_call(metrics_data[1])
        mock_get_metric.assert_any_call(metrics_data[2])
    
    def test_get_metrics_empty_list(self):
        # Test with empty list
        metrics = MetricFactory.get_metrics([])
        
        assert isinstance(metrics, list)
        assert len(metrics) == 0 