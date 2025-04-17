from abc import abstractmethod

import pandas as pd

from benchmarq.results import BenchmarkResult


class BaseBenchmark:
    @abstractmethod
    def grade_all(self, df: pd.DataFrame) -> BenchmarkResult:
        raise NotImplementedError()
