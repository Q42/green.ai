import json
import os
import subprocess
from abc import abstractmethod

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric, GEval, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from pydantic import BaseModel, ConfigDict


class Evaluator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def evaluate_test_case(self, input: Golden) -> LLMTestCase:
        pass

    @abstractmethod
    def evaluate_consumption(self, input: Golden):
        pass


def get_params(input: list[str]) -> list[LLMTestCaseParams]:
    return [LLMTestCaseParams(val) for val in input]


class MetricFactory(BaseModel):

    @staticmethod
    def get_metric(data: dict) -> BaseMetric:
        match data["type"]:
            case "GEval":
                data.pop("type")
                return GEval(**data)
            case "AnswerRelevancy":
                data.pop("type")
                return AnswerRelevancyMetric(**data)
            case "Faithfulness":
                data.pop("type")
                return FaithfulnessMetric(**data)
            case _:
                data.pop("type")
                return GEval(**data)

    @staticmethod
    def get_metrics_from_JSON(path: str) -> list[BaseMetric]:

        with open(os.path.join(os.path.dirname(__file__) , '..', path), "r") as f:
            data = json.loads(f.read())
            metrics = data["metrics"]
            metrics_dict: list[BaseMetric] = []
            for metric in metrics:
                metric["evaluation_params"] = get_params(metric["evaluation_params"])
                metrics_dict.append(MetricFactory.get_metric(metric))
            return metrics_dict

def serve_vllm(model: str) -> bool:
    subprocess.Popen(["vllm", "serve", "--model", model])
    return True