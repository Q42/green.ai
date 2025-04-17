
__all__ = [
    'get_dataset',
    'run',
    'evaluate_dataset',
    'export_results'
]

__app_name__ = "benchmarq"
__version__ = "0.1.0"

from benchmarq.benchmark import get_dataset, run, evaluate_dataset, export_results
