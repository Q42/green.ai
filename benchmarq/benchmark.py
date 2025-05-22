import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Callable, Any, Awaitable, List, Tuple

import pandas as pd
from codecarbon import EmissionsTracker
from tqdm.asyncio import tqdm

from benchmarq.benchmarks.MRCR import MRCR

from benchmarq.benchmarks.multi_challenge import MultiChallenge
from benchmarq.benchmarks.base import BaseBenchmark
from benchmarq.results import BenchmarkResult, ConsumptionResult


def get_dataset(config: dict) -> pd.DataFrame:
    df: pd.DataFrame
    match config["type"]:
        case "csv":
            df = pd.read_csv(Path(__file__).parent.parent / config["path"])
        case _:
            raise ValueError(f"Unknown benchmark type: {config['type']}")

    #if bool(os.environ["debug"]):
    #    df = df.head(2)

    return df


async def run(
        dataset: pd.DataFrame,
        func: Callable[[Any], Awaitable[str]]
) -> Tuple[pd.DataFrame, ConsumptionResult]:

    tracker = EmissionsTracker(
        tracking_mode="machine",
        experiment_id=uuid.uuid4().hex,
        log_level="error",
    )

    # Create a new 'response' column initialized with None
    dataset['response'] = None

    # Create a list to store tasks and their corresponding indices
    tasks: List[asyncio.Task] = []
    indices = []

    tracker.start()

    # Create tasks for each row
    for index, row in dataset.iterrows():
        task = asyncio.create_task(func(row))
        tasks.append(task)
        indices.append(index)

    # Gather all results with progress bar
    results = await tqdm.gather(*tasks)

    tracker.stop()

    # Assign results back to the dataframe
    for index, result in zip(indices, results):
        dataset.at[index, 'response'] = result

    return dataset, ConsumptionResult.from_tracker(tracker.final_emissions_data)


def __get_benchmark(benchmark_name: str) -> BaseBenchmark:
    match benchmark_name:
        case "MRCR":
            return MRCR()
        case "multi-challenge":
            return MultiChallenge()
        case _:
            return MRCR()


def evaluate_dataset(dataset: pd.DataFrame, config: dict) -> BenchmarkResult | None:
    if not config["accuracy"]:
        return None

    benchmark = __get_benchmark(config["benchmark"])

    return benchmark.grade_all(dataset)


def export_results(
        name: str,
        metadata: dict,
        config: dict,
        consumption:
        ConsumptionResult,
        accuracy: BenchmarkResult = None) -> None:

    base_dir = Path(__file__).parent.parent
    results_file_path = base_dir / "results" / f"{name}.json"

    result = {
        "name": name,
        **metadata,
        **(vars(accuracy) if accuracy is not None else {}),
        **vars(consumption),
        "config": config
    }

    if results_file_path.exists():
        with open(results_file_path, 'r') as f:
            data = json.load(f)
            data.append(result)

        with open(results_file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        results_file_path.parent.mkdir(exist_ok=True)

        data = [result]
        with open(results_file_path, 'w') as f:
            json.dump(data, f, indent=4)
