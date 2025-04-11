import subprocess
from threading import Lock
from typing import Optional, List, Dict, Union
from typing_extensions import TypedDict
from deepeval.metrics import BaseMetric, GEval, AnswerRelevancyMetric, FaithfulnessMetric, SummarizationMetric, \
    BaseConversationalMetric
from deepeval.test_case import LLMTestCaseParams
from pydantic import BaseModel

from benchmarq.custom_metrics.conversation_coherence.conversation_coherence import ConversationCoherenceMetric


def get_params(input: list[str]) -> list[LLMTestCaseParams]:
    return [LLMTestCaseParams(val) for val in input]


class MetricDict(TypedDict, total=False):
    type: str
    name: Optional[str]
    criteria: Optional[str]
    evaluation_params: Optional[List[str]]
    threshold: Optional[float]


class SettingsDict(TypedDict):
    datasets: Dict[str, str]
    metrics: List[MetricDict]


class MetricFactory(BaseModel):

    @staticmethod
    def get_metric(data: MetricDict) -> Union[BaseMetric, BaseConversationalMetric]:
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
            case "Summarization":
                data.pop("type")
                return SummarizationMetric(**data)
            case "ConversationalCoherence":
                data.pop("type")
                return ConversationCoherenceMetric(**data)
            case _:
                raise Exception("Metric type not found in MetricFactory. Please check the metric type in the JSON file.")

    @staticmethod
    def get_metrics(data: List[MetricDict]) -> List[BaseMetric]:
        metrics: List[BaseMetric] = []
        for metric in data:
            metrics.append(MetricFactory.get_metric(metric))
        return metrics


class VLLMServerSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VLLMServerSingleton, cls).__new__(cls)
                cls._instance._process = None
        return cls._instance

    def start_server(self, model: str):
        if self._process is not None:
            print("Server is already running.")
            return

        # Start the subprocess and capture the output
        self._process = subprocess.Popen(
            ["vllm", "serve", model],
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
