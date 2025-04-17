import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union, List, Optional

from codecarbon.output import EmissionsData


@dataclass
class ConsumptionResult:
    """
    Output object containing run data
    """

    timestamp: str
    project_name: str
    run_id: str
    experiment_id: str
    duration: float
    emissions: float
    emissions_rate: float
    cpu_power: float
    gpu_power: float
    ram_power: float
    cpu_energy: float
    gpu_energy: float
    ram_energy: float
    energy_consumed: float
    country_name: str
    country_iso_code: str
    region: Optional[str]
    cloud_provider: str
    cloud_region: str
    os: str
    python_version: str
    codecarbon_version: str
    cpu_count: float
    cpu_model: str
    gpu_count: Optional[float]
    gpu_model: Optional[str]
    longitude: float
    latitude: float
    ram_total_size: float
    tracking_mode: str
    on_cloud: str = "N"
    pue: float = 1

    @property
    def values(self) -> OrderedDict:
        return OrderedDict(self.__dict__.items())

    def compute_delta_emission(self, previous_emission):
        delta_duration = self.duration - previous_emission.duration
        self.duration = delta_duration
        delta_emissions = self.emissions - previous_emission.emissions
        self.emissions = delta_emissions
        self.cpu_energy -= previous_emission.cpu_energy
        self.gpu_energy -= previous_emission.gpu_energy
        self.ram_energy -= previous_emission.ram_energy
        self.energy_consumed -= previous_emission.energy_consumed
        if delta_duration > 0:
            # emissions_rate in g/s : delta_emissions in kg.CO2 / delta_duration in s
            self.emissions_rate = delta_emissions / delta_duration
        else:
            self.emissions_rate = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @classmethod
    def from_tracker(cls, instance: EmissionsData):
        return cls(
            timestamp=instance.timestamp,
            project_name=instance.project_name,
            run_id=instance.run_id,
            duration=instance.duration,
            emissions=instance.emissions,
            emissions_rate=instance.emissions_rate,
            cpu_power=instance.cpu_power,
            gpu_power=instance.gpu_power,
            ram_power=instance.ram_power,
            cpu_energy=instance.cpu_energy,
            gpu_energy=instance.gpu_energy,
            ram_energy=instance.ram_energy,
            energy_consumed=instance.energy_consumed,
            country_name=instance.country_name,
            country_iso_code=instance.country_iso_code,
            region=None if instance.region is float('nan') else instance.region,
            cloud_provider=instance.cloud_provider,
            cloud_region=instance.cloud_region,
            os=instance.os,
            python_version=instance.python_version,
            codecarbon_version=instance.codecarbon_version,
            cpu_count=instance.cpu_count,
            cpu_model=instance.cpu_model,
            gpu_count=instance.gpu_count,
            gpu_model=instance.gpu_model,
            longitude=instance.longitude,
            latitude=instance.latitude,
            ram_total_size=instance.ram_total_size,
            tracking_mode=instance.tracking_mode,
            on_cloud=instance.on_cloud,
            experiment_id="1",
        )


@dataclass
class BenchmarkResult:
    name: str
    score: float
    std: Optional[float]
    individual_score: List[Union[bool, float]]
