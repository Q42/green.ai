import json
import os
import uuid
from dataclasses import asdict
from typing import Any

from codecarbon import EmissionsTracker
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import TestResult
from deepeval.metrics import BaseMetric
from pydantic import BaseModel, Field, ConfigDict

from benchmarq.utility import Evaluator, MetricFactory
from benchmarq.results import RunResult, ConsumptionResult


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    subquestion_id: str
    subquestion_path: str
    name: str = Field(max_length=10)
    description: str = Field(min_length=15)
    settings: Evaluator
    metrics: list[BaseMetric] = Field(default_factory=list)
    runs: list[RunResult] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.metrics = MetricFactory.get_metrics_from_JSON(self.subquestion_path)

    def __consumption_test(self) -> ConsumptionResult:
        # setup
        tracker = EmissionsTracker(
            tracking_mode="machine",
            experiment_id=uuid.uuid4().hex,
        )

        # warmup
        for _ in range(2):
            self.settings.evaluate_consumption(Golden(input="something"))

        # test
        tracker.start()
        for _ in range(5):
            self.settings.evaluate_consumption(Golden(input="something"))
        tracker.stop()

        return ConsumptionResult.from_tracker(tracker.final_emissions_data)

    def __metric_test(self, dataset: EvaluationDataset) -> list[TestResult]:
        return dataset.evaluate(metrics=self.metrics).test_results

    def __test_dataset(self) -> EvaluationDataset:
        with open(os.path.join(os.path.dirname(__file__) , '..', self.subquestion_path,), "r") as f:
            data = json.loads(f.read())
            dataset_path = data["dataset"]["path"]
        dataset = EvaluationDataset()
        dataset.add_goldens_from_csv_file(
            file_path= str(os.path.join(os.path.dirname(__file__) , '..', dataset_path,)),
            input_col_name="input",
            actual_output_col_name="actual_output",
            expected_output_col_name="expected_output",
            context_col_name="context",
            retrieval_context_col_name="retrieval_context",
        )

        for golden in dataset.goldens:
            dataset.add_test_case(self.settings.evaluate_test_case(input=golden))

        return dataset

    def create_subquestion_json(self) -> dict:
        data = {
            'subquestion_id': self.subquestion_id,
            'subquestion_metrics_path': self.subquestion_path,
            'experiments': [
                {
                    'id': self.id,
                    'name': self.name,
                    'description': self.description,
                    'settings': self.settings.model_dump(),
                    'runs': [json.loads(run.toJSON()) for run in self.runs ],
                }
            ]
        }

        with open(f'{os.path.dirname(__file__)}/../results/{self.subquestion_id}.json', 'w') as f:
            json.dump(data, f)

        return data


    def __add_to_json(self):
        with open(f'{os.path.dirname(__file__)}/../results/{self.subquestion_id}.json', 'r') as f:
            data = json.load(f)
            data["experiments"]

    def __results_exist(self) -> bool:
        return os.path.isfile(f'{os.path.dirname(__file__)}/../results/{self.subquestion_id}.json')

    def run(self) -> RunResult:
        c_result: ConsumptionResult = self.__consumption_test()
        dataset = self.__test_dataset()
        m_result = self.__metric_test(dataset)
        result = RunResult(consumption_results=c_result, metric_results=m_result)
        self.runs.append(result)
        if (self.__results_exist()):
            self.__add_to_json()
        else:
            self.create_subquestion_json()
        return result
