"""Hardware system specifications for roofline analysis."""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class TflopsConfig:
  """Compute throughput specifications per datatype."""

  # Maps dtype string (e.g., 'bfloat16', 'float32', 'int8') to peak TFLOPS/TOPS
  peak_tflops_per_dtype: Dict[str, float]


@dataclass
class IciConfig:
  """Inter-Chip Interconnect specifications."""

  peak_bw_gbps: float
  bidirectional: bool


@dataclass
class HbmConfig:
  """High Bandwidth Memory specifications."""

  # List of tuples: (transfer_size_bytes, bandwidth_gb_s)
  curve_gbps: List[Tuple[int, float]]


@dataclass
class SystemConfig:
  """System hardware specifications."""

  name: str
  tflops: TflopsConfig
  ici: IciConfig
  hbm: HbmConfig


# TPU v7x (Ironwood / Ghostfish / GFC)
IRONWOOD = SystemConfig(
    name="ironwood",
    tflops=TflopsConfig(
        peak_tflops_per_dtype={
            "bfloat16": 2307.0,
            "float32": 1153.5,  # Estimated based on VPU capability
            "float8_e5m2": 4614.0,
            "float8_e4m3fn": 4614.0,
            "int8": 4614.0,
        }
    ),
    ici=IciConfig(
        peak_bw_gbps=1200.0,
        bidirectional=True,
    ),
    hbm=HbmConfig(
        curve_gbps=[
            (1024, 100.0),
            (1048576, 2000.0),
            (104857600, 5000.0),
            (1073741824, 7380.0),  # ~1GB transfer reaches peak 7380 GB/s
        ]
    ),
)

SYSTEMS = {
    "ironwood": IRONWOOD,
    "gfc": IRONWOOD,  # Alias for Ghostfish/Ironwood
}


def get_system(name: str) -> SystemConfig:
  if name.lower() not in SYSTEMS:
    raise ValueError(
        f"System {name} not found. Available: {list(SYSTEMS.keys())}"
    )
  return SYSTEMS[name.lower()]
