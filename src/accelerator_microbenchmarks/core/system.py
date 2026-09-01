"""Hardware system specifications for roofline analysis."""

import dataclasses
import importlib.metadata
from typing import Any
import jax


@dataclasses.dataclass
class TflopsConfig:
  """Compute throughput specifications per datatype."""

  # Maps dtype string (e.g., 'bfloat16', 'float32', 'int8') to peak TFLOPS/TOPS
  peak_tflops_per_dtype: dict[str, float]


@dataclasses.dataclass
class IciConfig:
  """Inter-Chip Interconnect specifications."""

  peak_bw_gbps: float
  bidirectional: bool


@dataclasses.dataclass
class HbmConfig:
  """High Bandwidth Memory specifications."""

  # List of tuples: (transfer_size_bytes, bandwidth_gb_s)
  curve_gbps: list[tuple[int, float]]


@dataclasses.dataclass
class SystemConfig:
  """System hardware specifications."""

  name: str
  topology_dimension: int = 3
  tflops: TflopsConfig | None = None
  ici: IciConfig | None = None
  hbm: HbmConfig | None = None


# TPU v7x (Ironwood / Ghostfish / GFC)
IRONWOOD = SystemConfig(
    name="ironwood",
    topology_dimension=3,
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

# TPU v6e (Trillium / Ghostlite / GLC)
TRILLIUM = SystemConfig(
    name="trillium",
    topology_dimension=2,
    tflops=TflopsConfig(
        peak_tflops_per_dtype={
            "bfloat16": 918.0,
            "float32": 459.0,
            "float8_e5m2": 918.0,
            "float8_e4m3fn": 918.0,
            "int8": 1836.0,
            "int4": 3672.0,
        }
    ),
    ici=IciConfig(
        peak_bw_gbps=800.0,
        bidirectional=True,
    ),
    hbm=HbmConfig(
        curve_gbps=[
            (1024, 50.0),
            (1048576, 800.0),
            (104857600, 1400.0),
            (1073741824, 1638.4),
        ]
    ),
)

SYSTEMS: dict[str, SystemConfig] = {
    "ironwood": IRONWOOD,
    "gfc": IRONWOOD,  # Alias for Ghostfish/Ironwood
    "v7": IRONWOOD,  # Alias for Ghostfish/Ironwood
    "tpu v7x": IRONWOOD,
    "tpu7x": IRONWOOD,
    "ghostlite": TRILLIUM,
    "v6e": TRILLIUM,  # Alias for Ghostlite/Trillium
    "trillium": TRILLIUM,  # Alias for Ghostlite/Trillium
    "glc": TRILLIUM,  # Alias for Ghostlite Core/Trillium
    "tpu v6 lite": TRILLIUM,
    "tpu v6": TRILLIUM,
}


def get_system(name: str) -> SystemConfig:
  if name.lower() not in SYSTEMS:
    raise ValueError(
        f"System {name} not found. Available: {list(SYSTEMS.keys())}"
    )
  return SYSTEMS[name.lower()]


def get_runtime_device_info() -> dict[str, Any]:
  """Extracts runtime environment details including JAX and LibTPU versions."""

  info = {
      "platform": str(jax.default_backend()),
      "device_count": jax.device_count(),
      "local_device_count": jax.local_device_count(),
      "jax_version": getattr(jax, "__version__", "unknown"),
  }

  try:
    try:
      info["libtpu_version"] = importlib.metadata.version("libtpu")
    except importlib.metadata.PackageNotFoundError:
      info["libtpu_version"] = importlib.metadata.version("libtpu-nightly")
  except Exception:
    pass

  try:
    info["chip_version"] = str(jax.devices()[0].device_kind)
  except Exception:
    pass

  return info
  # pylint: enable=g-import-not-at-top,broad-exception-caught
