import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, List, Dict

from codecarbon import EmissionsTracker
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import TestResult
from deepeval.metrics import BaseMetric
from pydantic import BaseModel, Field, ConfigDict, computed_field
from tqdm.asyncio import tqdm

from benchmarq.utility import Evaluator, MetricFactory
from benchmarq.results import RunResult, ConsumptionResult


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subquestion_id: str
    subquestion_path: str
    dataset_name: str
    dataset: EvaluationDataset = Field(default_factory=EvaluationDataset)
    name: str = Field(max_length=10)
    description: str = Field(min_length=15)
    settings: Evaluator
    metrics: List[BaseMetric] = Field(default_factory=list)
    runs: List[RunResult] = Field(default_factory=list)
    skip_metrics: bool = False
    debug_mode: bool = False
    tasks: List[asyncio.Task] = Field(default_factory=list)

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

    def __get_dataset(self) -> EvaluationDataset:
        with open(self.subquestion_file_path, "r") as f:
            data = json.load(f)
            dataset_path = next(
                (entry.get("path") for entry in data.get("dataset", [])
                 if entry.get("name") == self.dataset_name),
                None
            )
            if not dataset_path:
                raise ValueError(f"Dataset {self.dataset_name} not found in {self.subquestion_path}")
            self.dataset.goldens = []
            self.dataset.add_goldens_from_csv_file(
                file_path=str(self.base_dir / dataset_path),
                input_col_name="input",
                actual_output_col_name="actual_output",
                expected_output_col_name="expected_output",
                context_col_name="context",
                retrieval_context_col_name="retrieval_context",
            )


    def model_post_init(self, __context: Any) -> None:
        self.metrics = MetricFactory.get_metrics_from_JSON(self.subquestion_path)
        self.__get_dataset()

    async def __consumption_test(self) -> ConsumptionResult:
        # setup
        tracker = EmissionsTracker(
            tracking_mode="machine",
            experiment_id=uuid.uuid4().hex,
            log_level = "error",
        )

        # test
        print(f"testing: {self.name}")

        tracker.start()
        print(f"dataset len: {len(self.dataset.goldens)}")
        for row in self.dataset.goldens:
            task = asyncio.create_task(
                self.settings.async_evaluate_consumption(row)
            )
            self.tasks.append(task)
        await tqdm.gather(*self.tasks)

        tracker.stop()
        return ConsumptionResult.from_tracker(tracker.final_emissions_data)

    def __metric_test(self) -> List[TestResult]:
        return self.dataset.evaluate(metrics=self.metrics).test_results
            

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

        if self.debug_mode:
            dataset = EvaluationDataset(goldens=[self.dataset.goldens[0]])

        c_result: ConsumptionResult = await self.__consumption_test()
        m_result: List[TestResult] = []
        if not self.skip_metrics:
            for golden in self.dataset.goldens:
                self.dataset.add_test_case(await self.settings.evaluate_test_case(input=golden))
            m_result = self.__metric_test()
        result = RunResult(consumption_results=c_result, metric_results=m_result)
        self.runs.append(result)
        
        if self.__results_exist():
            self.__add_to_json(result)
        else:
            self.create_subquestion_json()
        return result
