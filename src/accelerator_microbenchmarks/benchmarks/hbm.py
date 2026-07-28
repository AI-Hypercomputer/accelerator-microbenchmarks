"""HBM bandwidth microbenchmarks supporting STREAM kernels (Copy, Scale, Add, Triad)."""

import dataclasses
import random
from typing import Any, Callable
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class HBMKernelSpec:
  """Specification for an HBM STREAM benchmark kernel.

  Attributes:
    name: Name of the kernel operation (e.g. copy, scale, add, triad).
    kernel_fn: Callable implementing the computation (receives tuple of args and
      scalar).
    num_inputs: Number of input arrays required.
    num_arrays: Total number of array transfers (reads + writes).
    num_flops_per_element: Multiplier for FLOP calculation (e.g. 1.0 or 2.0 for
      triad).
  """

  name: str
  kernel_fn: Callable[[tuple[Any, ...], Any], Any]
  num_inputs: int
  num_arrays: int
  num_flops_per_element: float = 1.0


HBM_KERNELS: dict[str, HBMKernelSpec] = {
    "copy": HBMKernelSpec(
        name="copy",
        kernel_fn=lambda args, scalar: args[0] + 1.0,
        num_inputs=1,
        num_arrays=2,  # 1 read (x), 1 write (y)
        num_flops_per_element=1.0,
    ),
    "scale": HBMKernelSpec(
        name="scale",
        kernel_fn=lambda args, scalar: args[0] * scalar,
        num_inputs=1,
        num_arrays=2,  # 1 read (x), 1 write (y)
        num_flops_per_element=1.0,
    ),
    "add": HBMKernelSpec(
        name="add",
        kernel_fn=lambda args, scalar: args[0] + args[1],
        num_inputs=2,
        num_arrays=3,  # 2 reads (x, y), 1 write (z)
        num_flops_per_element=1.0,
    ),
    "triad": HBMKernelSpec(
        name="triad",
        kernel_fn=lambda args, scalar: args[0] + (args[1] * scalar),
        num_inputs=2,
        num_arrays=3,  # 2 reads (x, y), 1 write (z)
        num_flops_per_element=2.0,
    ),
}


@registry.benchmark_registry.register("hbm_bandwidth")
class HBMBandwidthBenchmark(base.BaseBenchmark):
  """HBM bandwidth microbenchmark supporting standard STREAM kernels."""

  def __init__(self, mesh=None):
    super().__init__(mesh)
    self.spec: HBMKernelSpec | None = None
    self.scalar: Any | None = None

  def _resolve_spec(self, **params) -> HBMKernelSpec:
    """Resolve the active HBMKernelSpec from params or instance state."""
    if "op_type" in params:
      op_type = params["op_type"].lower()
      if op_type in HBM_KERNELS:
        return HBM_KERNELS[op_type]
      supported = ", ".join(HBM_KERNELS.keys())
      raise ValueError(
          f"Unsupported op_type: '{op_type}'. Supported: {supported}."
      )
    if self.spec is not None:
      return self.spec
    raise ValueError(
        "HBMBandwidthBenchmark is not configured. Call setup() before accessing"
        " benchmark properties or pass 'op_type' in params."
    )

  def setup(self, **params):
    op_type = params.get("op_type", "copy")
    spec = self._resolve_spec(op_type=op_type)
    self.spec = spec

    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str) if hasattr(jnp, dtype_str) else jnp.bfloat16
    self.scalar = jnp.array(random.uniform(1.1, 10.0), dtype=dtype)
    scalar = self.scalar

    @jax.jit
    def hbm_op(*args):
      with jax.named_scope(constants.MARKER):
        return spec.kernel_fn(args, scalar)

    self._jit_fn = hbm_op

  def get_run_identifier(self, **params) -> str:
    spec = self._resolve_spec(**params)
    size = params.get("size")
    if size is not None:
      return f"{spec.name}_dim_{size}"
    return f"{spec.name}"

  def generate_inputs(self, **params) -> tuple[jnp.ndarray, ...]:
    spec = self._resolve_spec(**params)
    size = params.get("size", 1024 * 1024 * 128)  # Default ~256MB for float16
    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str) if hasattr(jnp, dtype_str) else jnp.bfloat16

    # Force execution on local device 0 for single-device benchmark
    local_device = jax.local_devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(local_device)

    # Use jit with out_shardings to generate on device to avoid host OOM
    generate_data = jax.jit(
        lambda k: jax.random.normal(k, (size,), dtype=dtype),
        out_shardings=sharding,
    )

    inputs = []
    for i in range(spec.num_inputs):
      key = jax.random.PRNGKey(i)
      inputs.append(generate_data(key))
    return tuple(inputs)

  def run_op(self, *args, **kwargs) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args, **kwargs)

  def get_total_bytes(self, **params) -> float:
    spec = self._resolve_spec(**params)
    size = params.get("size", 1024 * 1024 * 128)
    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str) if hasattr(jnp, dtype_str) else jnp.bfloat16
    itemsize = jnp.dtype(dtype).itemsize

    return float(size * itemsize * spec.num_arrays)

  def get_arithmetic_intensity(self, **params) -> float:
    spec = self._resolve_spec(**params)
    size = params.get("size", 1024 * 1024 * 128)
    flops = float(spec.num_flops_per_element * size)
    bytes_moved = self.get_total_bytes(**params)
    return flops / bytes_moved

  def calculate_metrics(
      self, times_ms: list[float], **params
  ) -> dict[str, Any]:
    spec = self._resolve_spec(**params)
    metrics = super().calculate_metrics(times_ms, **params)
    total_bytes = self.get_total_bytes(**params)

    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = (total_bytes / avg_latency_s) / 1e9

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["total_bytes_mb"] = total_bytes / 1e6
    metrics["intensity"] = self.get_arithmetic_intensity(**params)
    metrics["op_type"] = spec.name
    return metrics
