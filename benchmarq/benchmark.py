import pandas as pd

from benchmarq.benchmarks.MRCR import MRCR
from benchmarq.benchmarks.base import BaseBenchmark
from benchmarq.results import BenchmarkResult

def get_benchmark(benchmark_name: str) -> BaseBenchmark:
    match benchmark_name:
        case "MRCR":
            return MRCR()
        case _:
            return MRCR()

def evaluate_dataset(dataset: pd.DataFrame, config: dict) -> BenchmarkResult | None:
    if not config["accuracy"]:
        return None

    benchmark = get_benchmark(config["benchmark"])

    return benchmark.grade_all(dataset)