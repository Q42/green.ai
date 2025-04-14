import asyncio
import json
import uuid
import pandas as pd
from pathlib import Path
from typing import Any, List, Dict, Callable, Awaitable, Union

from codecarbon import EmissionsTracker
from deepeval.dataset import EvaluationDataset, Golden, ConversationalGolden
from deepeval.evaluate import TestResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from pydantic import BaseModel, Field, ConfigDict, model_validator
from tqdm.asyncio import tqdm

from benchmarq.utility import MetricFactory, SettingsDict
from benchmarq.results import RunResult, ConsumptionResult


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    subquestion_id: str
    settings: SettingsDict
    dataset_name: str
    dataset: EvaluationDataset = Field(default_factory=EvaluationDataset)
    name: str = Field(max_length=10)
    description: str = Field(min_length=15)
    c_func: Callable[[Union[Golden, ConversationalGolden]], Awaitable[Union[LLMTestCase, ConversationalTestCase]]]
    a_func: Callable[[Union[Golden, ConversationalGolden]], Awaitable[Union[LLMTestCase, ConversationalTestCase]]] = None
    metrics: List[BaseMetric] = Field(default_factory=list)
    runs: List[RunResult] = Field(default_factory=list)
    skip_metrics: bool = False
    debug_mode: bool = False
    conversational: bool = False
    tasks: List[asyncio.Task] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        """Base directory for all path operations."""
        return Path(__file__).parent.parent

    @property
    def results_file_path(self) -> Path:
        """Path to the results JSON file."""
        return self.base_dir / "results" / f"{self.subquestion_id}.json"

    @model_validator(mode='after')
    def set_a_func_default(self):
        # If a_func is None, set it to c_func
        if self.a_func is None:
            self.a_func = self.c_func
        return self

    def __get_dataset(self):

        dataset_path = self.settings['datasets'][self.dataset_name]

        if not dataset_path:
            raise ValueError(f"Dataset {self.dataset_name} not found in config file")
        self.dataset.goldens = []
        self.dataset.add_goldens_from_csv_file(
            file_path=str(self.base_dir / dataset_path),
            input_col_name="input",
            actual_output_col_name="actual_output",
            expected_output_col_name="expected_output",
            context_col_name="context",
            retrieval_context_col_name="retrieval_context",
        )

    def __get_conversational_dataset(self):
        dataset_path = self.settings['datasets'][self.dataset_name]
        if not dataset_path:
            raise ValueError(f"Dataset {self.dataset_name} not found in config file")
        self.dataset.conversational_goldens = []
        df = pd.read_csv(str(self.base_dir / dataset_path))

        for index, row in df.iterrows():
            data = json.loads(row['conversation'])
            goldens = []
            for i in range(0, len(data)-1, 2):
                user_input = data[i]['content']
                system_output = data[i + 1]['content']
                goldens.append(Golden(input=user_input, actual_output=system_output))
            goldens.append(Golden(input=data[-1]['content']))
            self.dataset.conversational_goldens.append(ConversationalGolden(turns=goldens))

    def model_post_init(self, __context: Any) -> None:
        self.metrics = MetricFactory.get_metrics(self.settings["metrics"])
        self.__get_conversational_dataset() if self.conversational else self.__get_dataset()

    async def __consumption_test(self) -> ConsumptionResult:
        # setup
        tracker = EmissionsTracker(
            tracking_mode="machine",
            experiment_id=uuid.uuid4().hex,
            log_level="error",
        )

        rows = self.dataset.conversational_goldens if self.conversational else self.dataset.goldens

        tracker.start()

        for row in rows:
            task = asyncio.create_task(
                self.c_func(row)
            )
            self.tasks.append(task)
        await tqdm.gather(*self.tasks)


        tracker.stop()
        return ConsumptionResult.from_tracker(tracker.final_emissions_data)

    def __metric_test(self) -> List[TestResult]:
        return self.dataset.evaluate(metrics=self.metrics).test_results

    def create_run_json(self, run: RunResult) -> Dict[str, Any]:
        base_dict = {
            'subquestion_id': self.subquestion_id,
            'id': self.id,
            'name': self.name,
            'description': self.description,
            **json.loads(run.toJSON())
        }
        # Update with metadata dictionary
        base_dict.update(self.metadata)
        return base_dict

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
            self.dataset = EvaluationDataset(goldens=[self.dataset.goldens[0]])

        c_result: ConsumptionResult = await self.__consumption_test()

        m_result: List[TestResult] = []
        if not self.skip_metrics:
            for golden in self.dataset.goldens:
                self.dataset.add_test_case(await self.a_func(golden))
            m_result = self.__metric_test()


        result = RunResult(consumption_results=c_result, metric_results=m_result)
        self.runs.append(result)

        if self.__results_exist():
            self.__add_to_json(result)
        else:
            self.create_subquestion_json()
        return result

    async def run_conversational(self) -> RunResult:
        if self.debug_mode:
            self.dataset = EvaluationDataset(conversational_goldens=[self.dataset.conversational_goldens[0]])

        c_result: ConsumptionResult = await self.__consumption_test()

        m_result: List[TestResult] = []
        if not self.skip_metrics:
            for golden in self.dataset.conversational_goldens:
                self.dataset.add_test_case(await self.a_func(golden))
            m_result = self.__metric_test()

        result = RunResult(consumption_results=c_result, metric_results=m_result)
        self.runs.append(result)

        if self.__results_exist():
            self.__add_to_json(result)
        else:
            self.create_subquestion_json()
        return result
