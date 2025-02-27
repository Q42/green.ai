import uuid

from codecarbon import EmissionsTracker
from deepeval.dataset import EvaluationDataset, Golden
from pydantic import BaseModel, Field

from benchmarq.adapter import Evaluator
from benchmarq.results import RunResult, MetricResult, ConsumptionResult


class Experiment(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    subquestionId: str
    name: str = Field(max_length=10)
    description: str = Field(min_length=15)
    settings: Evaluator
    runs: list[RunResult] = Field(default_factory=list)

    def __consumption_test(self) -> ConsumptionResult:
        # setup
        tracker = EmissionsTracker(
            tracking_mode="machine",
            experiment_id=uuid.uuid4().hex,
        )

        #warmup
        for i in range(2):
            self.settings.evaluate_consumption(Golden(input="something"))

        #test
        tracker.start()
        for i in range(10):
            self.settings.evaluate_consumption(Golden(input="something"))
        tracker.stop()

        return ConsumptionResult.from_tracker(tracker.final_emissions_data)

    def __metric_test(self) -> list[MetricResult]:
        pass

    def __get_dataset(self) -> EvaluationDataset:
        pass

    def run(self) -> RunResult:
        c_result: ConsumptionResult = self.__consumption_test()
        return RunResult(consumption_results=c_result)
