import asyncio
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Dict

from codecarbon import EmissionsTracker
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate import TestResult
from deepeval.metrics import BaseMetric
from pydantic import BaseModel, Field, ConfigDict

from benchmarq.utility import Evaluator, MetricFactory
from benchmarq.results import RunResult, ConsumptionResult


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subquestion_id: str
    subquestion_path: str
    dataset_name: str
    name: str = Field(max_length=10)
    description: str = Field(min_length=15)
    settings: Evaluator
    metrics: List[BaseMetric] = Field(default_factory=list)
    runs: List[RunResult] = Field(default_factory=list)

    @property
    def base_dir(self) -> Path:
        """Base directory for all path operations."""
        return Path(__file__).parent.parent

    @property
    def subquestion_file_path(self) -> Path:
        """Full path to the subquestion file."""
        return self.base_dir / self.subquestion_path

    @property
    def results_file_path(self) -> Path:
        """Path to the results JSON file."""
        return self.base_dir / "results" / f"{self.subquestion_id}.json"

    def model_post_init(self, __context: Any) -> None:
        self.metrics = MetricFactory.get_metrics_from_JSON(self.subquestion_path)

    async def __consumption_test(self, dataset: EvaluationDataset) -> ConsumptionResult:
        # setup
        tracker = EmissionsTracker(
            tracking_mode="machine",
            experiment_id=uuid.uuid4().hex,
        )
        tasks: List[asyncio.Task] = []
        
        # warmup
        for row in dataset.goldens:
            task = asyncio.create_task(
                self.settings.async_evaluate_consumption(row)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

        # test
        tracker.start()
        tasks = []
        for row in dataset.goldens:
            task = asyncio.create_task(
                self.settings.async_evaluate_consumption(row)
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

        tracker.stop()
        return ConsumptionResult.from_tracker(tracker.final_emissions_data)

    def __metric_test(self, dataset: EvaluationDataset) -> List[TestResult]:
        return dataset.evaluate(metrics=self.metrics).test_results

    def __test_dataset(self) -> EvaluationDataset:
        with open(self.subquestion_file_path, "r") as f:
            data = json.load(f)
            dataset_path = next(
                (entry.get("path") for entry in data.get("dataset", []) 
                 if entry.get("name") == self.dataset_name),
                None
            )
            
        if not dataset_path:
            raise ValueError(f"Dataset {self.dataset_name} not found in {self.subquestion_path}")

        dataset = EvaluationDataset()
        dataset.add_goldens_from_csv_file(
            file_path=str(self.base_dir / dataset_path),
            input_col_name="input",
            actual_output_col_name="actual_output",
            expected_output_col_name="expected_output",
            context_col_name="context",
            retrieval_context_col_name="retrieval_context",
        )

        for golden in dataset.goldens:
            dataset.add_test_case(self.settings.evaluate_test_case(input=golden))

        return dataset

    def create_run_json(self, run: RunResult) -> Dict[str, Any]:
        return {
            'subquestion_id': self.subquestion_id,
            'subquestion_metrics_path': self.subquestion_path,
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'settings': self.settings.model_dump(),
            **json.loads(run.toJSON())
        }

    def create_subquestion_json(self) -> List[Dict[str, Any]]:
        data = [self.create_run_json(run) for run in self.runs]
        self.results_file_path.parent.mkdir(exist_ok=True)
        
        with open(self.results_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return data

    def __add_to_json(self, run: RunResult) -> None:
        with open(self.results_file_path, 'r') as f:
            data = json.load(f)
            data.append(self.create_run_json(run))
        
        with open(self.results_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def __results_exist(self) -> bool:
        return self.results_file_path.exists()

    async def run(self) -> RunResult:
        dataset = self.__test_dataset()
        c_result: ConsumptionResult = await self.__consumption_test(dataset)

        m_result = self.__metric_test(dataset)
        result = RunResult(consumption_results=c_result, metric_results=m_result)
        self.runs.append(result)
        
        if self.__results_exist():
            self.__add_to_json(result)
        else:
            self.create_subquestion_json()
        return result
