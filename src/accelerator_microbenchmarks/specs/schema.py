"""Canonical TPU version identifiers and hardware specification data types.

This module is a leaf: it imports nothing else from this package, so every
per-generation spec module can depend on it without creating an import cycle
and `HardwareSpec.name` can be annotated with `TpuVersion` directly.

Concrete per-generation values live in sibling modules, one per TPU
generation, and `core.system` aggregates them into the `HARDWARE_SPECS`
mapping that the rest of the codebase consumes.
"""

import dataclasses
import enum

# The canonical native precision baseline for Google TPU MXU architectures.
DEFAULT_FALLBACK_DTYPE: str = "bfloat16"


class TpuVersion(str, enum.Enum):
  """Canonical TPU generation identifiers in TPUMS."""

  TPU7X = "tpu7x"
  V6E = "v6e"

  def __str__(self) -> str:
    return str(self.value)

  @classmethod
  def from_device_kind(cls, device_kind: "str | TpuVersion") -> "TpuVersion":
    """Maps a JAX PJRT `device_kind` to a canonical TPU version.

    Only `device_kind` spellings are recognized, matched case insensitively
    and ignoring surrounding whitespace. Conversions from other namespaces,
    such as marketing names or another framework's device strings, belong in
    their own classmethods so that the mappings cannot collide.

    Args:
      device_kind: A `jax.Device.device_kind` such as "TPU7x" or "TPU v6
        lite", or an already resolved `TpuVersion`.

    Returns:
      The `TpuVersion` member of the matching generation.

    Raises:
      ValueError: If `device_kind` does not name a supported generation.
    """
    if isinstance(device_kind, cls):
      return device_kind
    normalized = str(device_kind).strip().lower()
    if normalized in ("tpu7x", "tpu v7x"):
      return cls.TPU7X
    if normalized in ("v6e", "tpu v6e", "tpu v6 lite"):
      return cls.V6E
    raise ValueError(
        f"Unsupported TPU device_kind '{device_kind}'. Available versions:"
        f" {[version.value for version in cls]}"
    )


@dataclasses.dataclass(frozen=True)
class TflopsSpec:
  """Compute throughput specifications per datatype."""

  # Maps dtype string (e.g., 'bfloat16', 'float32', 'int8') to peak TFLOPS
  # per device (TensorCore). A value of `None` marks the dtype as natively
  # unsupported on this hardware, which disables compute-roofline metrics for
  # it. Dtypes absent from the mapping fall back to `DEFAULT_FALLBACK_DTYPE`.
  peak_tflops_per_device: dict[str, float | None]


class WraparoundRule(str, enum.Enum):
  """Rule deciding which ICI topology axes have wrap-around (torus) links.

  Attributes:
    NONE: No wrap-around links on any axis.
    ALL_DIMS_GE_4: 3D torus; every axis wraps iff all topology dims are >= 4
      (e.g. 4x4x4, 8x8x8). Topologies such as 2x4x4 or 4x4x1 have no wrap.
    PER_DIM_GE_16: 2D; an axis wraps iff its topology dim is >= 16 (e.g.
      16x16, or only the 16-axis of 16x8).
  """

  NONE = "none"
  ALL_DIMS_GE_4 = "all_dims_ge_4"
  PER_DIM_GE_16 = "per_dim_ge_16"


@dataclasses.dataclass(frozen=True)
class IciSpec:
  """Inter-Chip Interconnect specification (theoretical line rate).

  Attributes:
    link_bw_per_direction_gb_s: Theoretical bandwidth of a single ICI link
      (port) in one direction, in decimal GB/s. Links are full-duplex, so each
      direction carries this bandwidth independently.
    wraparound_rule: Which topology axes have wrap-around links; doubles the
      bisection links on wrapped axes.
  """

  link_bw_per_direction_gb_s: float
  wraparound_rule: WraparoundRule = WraparoundRule.NONE


@dataclasses.dataclass(frozen=True)
class HbmSpec:
  """High Bandwidth Memory specifications (per-chip).

  For the classical roofline model, peak_bw_gb_s represents the flat
  asymptotic physical peak bandwidth ceiling (speed of light) from the
  hardware datasheet.
  """

  peak_bw_gb_s: float


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
  """Hardware accelerator specifications."""

  # Canonical TPU generation identifier, e.g. `TpuVersion.TPU7X`.
  name: TpuVersion
  topology_dimension: int = 3
  devices_per_chip: int = 1
  tflops: TflopsSpec | None = None
  ici: IciSpec | None = None
  hbm: HbmSpec | None = None

  @property
  def peak_hbm_bandwidth_per_device(self) -> float:
    """Returns the asymptotic peak HBM bandwidth for a single device (GB/s)."""
    if not self.hbm or self.devices_per_chip <= 0:
      return 0.0
    return self.hbm.peak_bw_gb_s / self.devices_per_chip
