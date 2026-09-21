"""Hardware system specifications for roofline analysis.

The concrete per-generation specifications live in the `specs` package, one
module per TPU generation. This module aggregates them into `HARDWARE_SPECS`
and is the stable entry point for the rest of the codebase, re-exporting the
spec data types so callers never import the `specs` package directly.
"""

from accelerator_microbenchmarks.specs import schema
from accelerator_microbenchmarks.specs import tpu7x
from accelerator_microbenchmarks.specs import v6e


TpuVersion = schema.TpuVersion
HardwareSpec = schema.HardwareSpec
TflopsSpec = schema.TflopsSpec
IciSpec = schema.IciSpec
HbmSpec = schema.HbmSpec

# The canonical native precision baseline for Google TPU MXU architectures.
DEFAULT_FALLBACK_DTYPE: str = schema.DEFAULT_FALLBACK_DTYPE

TPU7X_HARDWARE_SPEC: HardwareSpec = tpu7x.TPU7X_HARDWARE_SPEC
V6E_HARDWARE_SPEC: HardwareSpec = v6e.V6E_HARDWARE_SPEC

# Every generation available in this checkout, keyed by canonical
# `TpuVersion`. Adding a generation means adding its spec module and one entry
# here; generations excluded from a checkout, for example by a Copybara
# export, are absent from both `TpuVersion` and this mapping.
HARDWARE_SPECS: dict[TpuVersion, HardwareSpec] = {
    TpuVersion.TPU7X: TPU7X_HARDWARE_SPEC,
    TpuVersion.V6E: V6E_HARDWARE_SPEC,
}


def get_hardware_spec(version: TpuVersion) -> HardwareSpec:
  """Retrieves the hardware spec of a canonical TPU generation.

  Args:
    version: The canonical TPU generation, for example `TpuVersion.TPU7X`.
      Resolve a JAX `device_kind` with `TpuVersion.from_device_kind` first.

  Returns:
    The hardware spec of the requested generation.

  Raises:
    ValueError: If no spec is available for `version` in this checkout.
  """
  if version not in HARDWARE_SPECS:
    raise ValueError(
        f"Unsupported TPU hardware '{version}'. Available versions:"
        f" {sorted(str(available) for available in HARDWARE_SPECS)}"
    )
  return HARDWARE_SPECS[version]
