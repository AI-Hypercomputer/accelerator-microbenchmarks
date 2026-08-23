"""HBM bandwidth microbenchmarks supporting STREAM kernels (Copy, Scale, Add, Triad)."""

import dataclasses
import random
from typing import Any, Callable
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import utils
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


@dataclasses.dataclass
class HBMBandwidthParams(base.BaseBenchmarkParams):
  op_type: str = "copy"
  size: int = 134217728  # default ~256MB for bfloat16 (128M elements * 2 bytes)
  dtype: str = "bfloat16"
  device_id: int = 0


@registry.benchmark_registry.register("hbm", aliases=["hbm_bandwidth"])
class HBMBandwidthBenchmark(base.BaseBenchmark[HBMBandwidthParams]):
  Config = HBMBandwidthParams
  """HBM bandwidth microbenchmark supporting standard STREAM kernels."""

  def __init__(
      self, config: HBMBandwidthParams, mesh: jax.sharding.Mesh | None = None
  ):
    super().__init__(config=config, mesh=mesh)
    self.spec: HBMKernelSpec | None = None
    self.scalar: Any | None = None

  def setup(self):
    num_devices = len(jax.devices())
    if not (0 <= self.config.device_id < num_devices):
      raise ValueError(
          f"Invalid device_id: {self.config.device_id}. Must be in range"
          f" [0, {num_devices - 1}] (found {num_devices} local device(s))."
      )

    op_type = self.config.op_type
    if op_type is None:
      raise ValueError("op_type must be specified.")

    op_type = op_type.lower()
    if op_type in HBM_KERNELS:
      spec = HBM_KERNELS[op_type]
    else:
      supported = ", ".join(HBM_KERNELS.keys())
      raise ValueError(
          f"Unsupported op_type: '{op_type}'. Supported: {supported}."
      )
    self.spec = spec

    dtype = utils.parse_dtype(self.config.dtype)
    self.scalar = jnp.array(random.uniform(1.1, 10.0), dtype=dtype)
    scalar = self.scalar

    @jax.jit
    def hbm_op(*args):
      with jax.named_scope(constants.MARKER):
        return spec.kernel_fn(args, scalar)

    self._jit_fn = hbm_op

  def get_run_identifier(self) -> str:
    return f"{self.config.op_type}_dim_{self.config.size}_dev_{self.config.device_id}"

  def get_device_to_measure(self) -> jax.Device:
    return jax.devices()[self.config.device_id]

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    assert self.spec is not None
    # 'size' being the number of elements
    size = self.config.size
    dtype = utils.parse_dtype(self.config.dtype)

    # Force execution on target local device for single-device benchmark
    device_to_measure = self.get_device_to_measure()
    sharding = jax.sharding.SingleDeviceSharding(device_to_measure)

    # Use jit with out_shardings to generate on device to avoid host OOM
    generate_data = jax.jit(
        lambda k: jax.random.normal(k, (size,), dtype=dtype),
        out_shardings=sharding,
    )

    inputs = []
    for i in range(self.spec.num_inputs):
      key = jax.random.PRNGKey(i)
      inputs.append(generate_data(key))
    return tuple(inputs)

  def run_op(self, *args, **kwargs) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args, **kwargs)

  def get_total_bytes(self) -> float:
    assert self.spec is not None
    size = self.config.size
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    return float(size * itemsize * self.spec.num_arrays)

  def get_arithmetic_intensity(self) -> float:
    assert self.spec is not None
    size = self.config.size
    flops = float(self.spec.num_flops_per_element * size)
    bytes_moved = self.get_total_bytes()
    return flops / bytes_moved

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    assert self.spec is not None
    metrics = super().calculate_metrics(times_ms)
    total_bytes = self.get_total_bytes()

    avg_latency_s = metrics["avg_ms"] / 1000.0
    if avg_latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = (total_bytes / avg_latency_s) / 1e9

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["total_bytes_mb"] = total_bytes / 1e6
    metrics["intensity"] = self.get_arithmetic_intensity()
    metrics["op_type"] = self.spec.name
    return metrics
