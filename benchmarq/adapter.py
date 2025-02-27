from abc import abstractmethod

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ConfigDict


class Evaluator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def evaluate_test_case(self, input: Golden) -> LLMTestCase:
        pass

    @abstractmethod
    def evaluate_consumption(self, input: Golden):
        pass