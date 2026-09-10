"""Hardware system specifications for roofline analysis."""

import dataclasses
from typing import Any
import enum


class TpuVersion(str, enum.Enum):
  """Canonical TPU generation identifiers in TPUMS."""

  TPU7X = "tpu7x"
  V6E = "v6e"

  def __str__(self) -> str:
    return str(self.value)

  @classmethod
  def from_str(cls, val: Any) -> "TpuVersion":
    """Normalizes arbitrary string, device_kind, or enum to canonical TpuVersion."""
    if isinstance(val, cls):
      return val
    low = str(val).strip().lower() if val else ""
    if "7" in low or "ironwood" in low:
      return cls.TPU7X
    if "v6e" in low or "6e" in low or "v6 lite" in low or "trillium" in low:
      return cls.V6E
    raise ValueError(
        f"Unsupported TPU hardware '{val}'. Available versions:"
        f" {[v.value for v in cls]}"
    )


# The canonical native precision baseline for Google TPU MXU architectures.
DEFAULT_FALLBACK_DTYPE: str = "bfloat16"


@dataclasses.dataclass(frozen=True)
class TflopsSpec:
  """Compute throughput specifications per datatype."""

  # Maps dtype string (e.g., 'bfloat16', 'float32', 'int8') to peak TFLOPS
  # per device (TensorCore).
  peak_tflops_per_device: dict[str, float]


@dataclasses.dataclass(frozen=True)
class IciSpec:
  """Inter-Chip Interconnect specifications (per-chip)."""

  peak_bw_gbps: float
  bidirectional: bool

  @property
  def peak_bw_gbps_per_chip(self) -> float:
    return self.peak_bw_gbps


@dataclasses.dataclass(frozen=True)
class HbmSpec:
  """High Bandwidth Memory specifications (per-chip)."""

  # List of tuples: (transfer_size_bytes, bandwidth_per_chip_gb_s) representing
  # the total per-chip HBM bandwidth.
  curve_gbps: list[tuple[int, float]]

  @property
  def curve_gbps_per_chip(self) -> list[tuple[int, float]]:
    return self.curve_gbps

  @property
  def peak_bandwidth_per_chip(self) -> float:
    """Returns the asymptotic peak HBM bandwidth per chip (GB/s)."""
    if not self.curve_gbps:
      return 0.0
    return max(bw for _, bw in self.curve_gbps)

  def get_curve_gbps_per_device(
      self, devices_per_chip: int = 1
  ) -> list[tuple[int, float]]:
    """Returns the HBM bandwidth curve scaled to per-device (TensorCore)."""
    return [(size, bw / devices_per_chip) for size, bw in self.curve_gbps]


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
  """Hardware accelerator specifications."""

  name: TpuVersion
  topology_dimension: int = 3
  devices_per_chip: int = 1
  tflops: TflopsSpec | None = None
  ici: IciSpec | None = None
  hbm: HbmSpec | None = None

  def get_hbm_curve_per_device(self) -> list[tuple[int, float]] | None:
    """Returns HBM bandwidth curve per device based on devices_per_chip."""
    if not self.hbm:
      return None
    return self.hbm.get_curve_gbps_per_device(self.devices_per_chip)

  @property
  def peak_hbm_bandwidth_per_device(self) -> float:
    """Returns the asymptotic peak HBM bandwidth for a single device (GB/s)."""
    if not self.hbm:
      return 0.0
    return self.hbm.peak_bandwidth_per_chip / self.devices_per_chip


# TPU v7x (Ironwood)
TPU7X_HARDWARE_SPEC = HardwareSpec(
    name=TpuVersion.TPU7X,
    topology_dimension=3,
    devices_per_chip=2,
    tflops=TflopsSpec(
        peak_tflops_per_device={
            "bfloat16": 1153.5,
            "float32": 576.75,  # Estimated based on VPU capability
            "float8_e5m2": 2307.0,
            "float8_e4m3fn": 2307.0,
            "int8": 2307.0,
        }
    ),
    ici=IciSpec(
        peak_bw_gbps=1200.0,
        bidirectional=True,
    ),
    hbm=HbmSpec(
        curve_gbps=[
            (1024, 100.0),
            (1048576, 2000.0),
            (104857600, 5000.0),
            (1073741824, 7380.0),  # ~1GB transfer reaches peak 7380 GB/s
        ]
    ),
)

# TPU v6e (Trillium)
V6E_HARDWARE_SPEC = HardwareSpec(
    name=TpuVersion.V6E,
    topology_dimension=2,
    devices_per_chip=1,
    tflops=TflopsSpec(
        peak_tflops_per_device={
            "bfloat16": 918.0,
            "float32": 459.0,
            "float8_e5m2": 918.0,
            "float8_e4m3fn": 918.0,
            "int8": 1836.0,
            "int4": 3672.0,
        }
    ),
    ici=IciSpec(
        peak_bw_gbps=800.0,
        bidirectional=True,
    ),
    hbm=HbmSpec(
        curve_gbps=[
            (1024, 50.0),
            (1048576, 800.0),
            (104857600, 1400.0),
            (1073741824, 1638.4),
        ]
    ),
)

HARDWARE_SPECS: dict[TpuVersion, HardwareSpec] = {
    TpuVersion.TPU7X: TPU7X_HARDWARE_SPEC,
    TpuVersion.V6E: V6E_HARDWARE_SPEC,
}


def get_hardware_spec(target: TpuVersion | str) -> HardwareSpec:
  """Retrieves a HardwareSpec by TpuVersion enum or string alias."""
  tpu_version = TpuVersion.from_str(target)
  if tpu_version not in HARDWARE_SPECS:
    raise ValueError(
        f"Hardware spec for '{tpu_version}' not found in HARDWARE_SPECS"
        " registry."
    )
  return HARDWARE_SPECS[tpu_version]

