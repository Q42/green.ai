from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from benchmarq.benchmarks.base import BaseBenchmark
from benchmarq.results import BenchmarkResult


def _grade(row) -> float:
    """
    Compare response and answer.
    """
    if not row['response'].startswith(row['random_string_to_prepend']):
        return 0
    response = row['response'].removeprefix(row['random_string_to_prepend'])
    answer = row['answer'].removeprefix(row['random_string_to_prepend'])
    return float(SequenceMatcher(None, response, answer).ratio())


class MRCR(BaseBenchmark):

    def grade_all(self, df: pd.DataFrame) -> BenchmarkResult:
        scores = []

        for index, row in df.iterrows():
            scores.append(_grade(row))

        return BenchmarkResult(name="MRCR",score=float(np.mean(scores)), std=float(np.std(scores)), individual_score=scores)
