import pytest
from typing import Dict, List, cast
from benchmarq.utility import MetricDict, SettingsDict


class TestTypedDicts:
    def test_metric_dict_minimal(self):
        # Test with minimal required fields
        metric: MetricDict = {"type": "GEval"}
        
        assert metric["type"] == "GEval"
        assert "name" not in metric
        assert "criteria" not in metric
        assert "evaluation_params" not in metric
        assert "threshold" not in metric
    
    def test_metric_dict_full(self):
        # Test with all fields
        metric: MetricDict = {
            "type": "GEval",
            "name": "test metric",
            "criteria": "test criteria",
            "evaluation_params": ["param1", "param2"],
            "threshold": 0.75
        }
        
        assert metric["type"] == "GEval"
        assert metric["name"] == "test metric"
        assert metric["criteria"] == "test criteria"
        assert metric["evaluation_params"] == ["param1", "param2"]
        assert metric["threshold"] == 0.75
    
    def test_metric_dict_missing_required(self):
        # Test with missing required field
        with pytest.raises(KeyError):
            metric: MetricDict = {}
            metric_type = metric["type"]
    
    def test_settings_dict(self):
        # Test SettingsDict
        settings: SettingsDict = {
            "datasets": {
                "test_dataset": "/path/to/dataset",
                "another_dataset": "/path/to/another"
            },
            "metrics": [
                {"type": "GEval", "name": "test1"},
                {"type": "AnswerRelevancy", "threshold": 0.8}
            ]
        }
        
        assert len(settings["datasets"]) == 2
        assert settings["datasets"]["test_dataset"] == "/path/to/dataset"
        assert settings["datasets"]["another_dataset"] == "/path/to/another"
        
        assert len(settings["metrics"]) == 2
        assert settings["metrics"][0]["type"] == "GEval"
        assert settings["metrics"][0]["name"] == "test1"
        assert settings["metrics"][1]["type"] == "AnswerRelevancy"
        assert settings["metrics"][1]["threshold"] == 0.8
    
    def test_settings_dict_missing_field(self):
        # Test with missing field
        with pytest.raises(KeyError):
            settings: SettingsDict = {
                "datasets": {"test": "path"}
                # Missing "metrics" field
            }
            metrics = settings["metrics"]
    
    # Note: TypedDict doesn't actually enforce type checking at runtime
    # Python's structural typing for dictionaries doesn't validate types at runtime
    # We can only test that dictionaries with the right structure are accepted
    # and that required keys are enforced
    
    def test_settings_dict_runtime_behavior(self):
        # In reality, Python will accept these at runtime because 
        # TypedDict is just a type hint and doesn't enforce runtime validation
        
        # This is valid syntax in Python, though not type-safe
        settings1 = {
            "datasets": ["path1", "path2"],  # Should be Dict[str, str]
            "metrics": []
        }
        
        # This is also valid syntax in Python
        settings2 = {
            "datasets": {"test": "path"},
            "metrics": {"type": "GEval"}  # Should be List[MetricDict]
        }
        
        # Test that we can access these fields (even though they're wrong types)
        assert settings1["datasets"] == ["path1", "path2"]
        assert settings2["metrics"] == {"type": "GEval"}
        
        # For documentation purposes, we'll show what would be correct
        valid_settings: SettingsDict = {
            "datasets": {"test": "path"},
            "metrics": [{"type": "GEval"}]
        } 