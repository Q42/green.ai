import json
import os
import subprocess
from abc import abstractmethod
from threading import Lock

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

class VLLMServerSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VLLMServerSingleton, cls).__new__(cls)
                cls._instance._process = None
        return cls._instance

    def start_server(self, model_name: str="facebook/opt-125m"):
        if self._process is not None:
            print("Server is already running.")
            return

        # Start the subprocess and capture the output
        self._process = subprocess.Popen(
            ["vllm", "serve", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Monitor the output line by line
        while True:
            output = self._process.stdout.readline()
            if output == "" and self._process.poll() is not None:
                break
            if "INFO:     Application startup complete." in output:
                print("Server started successfully.")
                return

        self._process = None
        print("Failed to start the server.")

    def stop_server(self):
        if self._process:
            self._process.terminate()  # Gracefully terminate the process
            try:
                self._process.wait(timeout=5)  # Wait for the process to terminate
            except subprocess.TimeoutExpired:
                self._process.kill()  # Forcefully kill the process if it doesn't terminate
            finally:
                self._process = None
                print("Server stopped.")
        else:
            print("No server is running.")