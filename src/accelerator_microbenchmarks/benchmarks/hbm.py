"""HBM bandwidth microbenchmarks supporting memory bandwidth kernels (Copy, Scale, Add, Triad)."""

import dataclasses
import random
from typing import Any, Callable, Sequence
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp
import numpy as np

# Mesh axis name used to shard HBM buffers across target devices.
_DEVICE_AXIS = "device"


class HBMKernelOp(constants.ParamEnum):
  """Operation types supported by HBM bandwidth microbenchmarks."""

  COPY = "copy"
  SCALE = "scale"
  ADD = "add"
  TRIAD = "triad"
  READ_ONLY = "read_only"
  WRITE_ONLY = "write_only"
  READ = "read"  # Alias for READ_ONLY.
  WRITE = "write"  # Alias for WRITE_ONLY.


@dataclasses.dataclass(frozen=True)
class HBMKernelSpec:
  """Specification for an HBM benchmark kernel.

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
  kernel_fn: Callable[..., Any]
  num_inputs: int
  num_arrays: int
  num_flops_per_element: float = 1.0


HBM_KERNELS: dict[HBMKernelOp, HBMKernelSpec] = {
    HBMKernelOp.COPY: HBMKernelSpec(
        name=HBMKernelOp.COPY.value,
        kernel_fn=lambda args, scalar: args[0] + jnp.ones((), args[0].dtype),
        num_inputs=1,
        num_arrays=2,  # 1 read (x), 1 write (y)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.SCALE: HBMKernelSpec(
        name=HBMKernelOp.SCALE.value,
        kernel_fn=lambda args, scalar: args[0] * scalar,
        num_inputs=1,
        num_arrays=2,  # 1 read (x), 1 write (y)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.ADD: HBMKernelSpec(
        name=HBMKernelOp.ADD.value,
        kernel_fn=lambda args, scalar: args[0] + args[1],
        num_inputs=2,
        num_arrays=3,  # 2 reads (x, y), 1 write (z)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.TRIAD: HBMKernelSpec(
        name=HBMKernelOp.TRIAD.value,
        kernel_fn=lambda args, scalar: args[0] + (args[1] * scalar),
        num_inputs=2,
        num_arrays=3,  # 2 reads (x, y), 1 write (z)
        num_flops_per_element=2.0,
    ),
    HBMKernelOp.READ_ONLY: HBMKernelSpec(
        name=HBMKernelOp.READ_ONLY.value,
        kernel_fn=lambda args, scalar: jnp.any(args[0], keepdims=True),
        num_inputs=1,
        num_arrays=1,  # 1 read (x), 0 writes (per-device boolean return)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.WRITE_ONLY: HBMKernelSpec(
        name=HBMKernelOp.WRITE_ONLY.value,
        kernel_fn=lambda args, scalar, shape, dtype: jnp.full(
            shape, scalar, dtype=dtype
        ),
        num_inputs=0,
        num_arrays=1,  # 0 reads, 1 write (y)
        num_flops_per_element=0.0,
    ),
}
HBM_KERNELS[HBMKernelOp.READ] = HBM_KERNELS[HBMKernelOp.READ_ONLY]
HBM_KERNELS[HBMKernelOp.WRITE] = HBM_KERNELS[HBMKernelOp.WRITE_ONLY]


@dataclasses.dataclass
class HBMBandwidthParams(
    base.ScopeParamsMixin, base.SingleDtypeBenchmarkParams
):
  """Parameters for HBM bandwidth microbenchmarks."""

  # bool copy (`x | True`) constant-folds and scale (`x & True`) is the
  # identity, so neither measures a real read + write of x. complex64 runs at
  # float32 latency while bytes are counted at 8 bytes per element, so it
  # reports ~2x the float32 bandwidth, above the HBM peak.
  DTYPE_CONSTRAINTS = {"dtype": (jnp.integer, jnp.floating)}
  # get_total_bytes() counts itemsize bytes per element.
  ALLOW_SUB_BYTE_DTYPES = False

  op_type: HBMKernelOp = dataclasses.field(
      default=HBMKernelOp.COPY,
      metadata={"help": "HBM kernel operation type."},
  )
  num_elements: int = dataclasses.field(
      default=134217728,
      metadata={
          "min": 1,
          "help": "Number of elements per device for HBM input arrays.",
      },
  )


@registry.benchmark_registry.register("hbm", aliases=["hbm_bandwidth"])
class HBMBandwidthBenchmark(base.BaseBenchmark[HBMBandwidthParams]):
  """HBM bandwidth microbenchmark supporting standard memory kernels."""

  Config = HBMBandwidthParams  # pylint: disable=invalid-name
  derive_chip_bandwidth: bool = False
  # Per-device HBM kernels; no inter-device (ICI) traffic.
  requires_ici: bool = False
  roofline_mode: constants.RooflineMode = constants.RooflineMode.MEMORY_HBM
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("op_type", report.format_str),
      ("scope", report.format_str),
      ("local_index", report.format_str),
      ("num_devices", report.format_str),
      ("num_elements", report.format_str),
      ("data_size_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("wall_clock_bandwidth_per_chip_gb_s", report.format_2f),
      ("wall_clock_bandwidth_per_host_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_device_gb_s", report.format_2f),
      ("xprof_bandwidth_per_chip_gb_s", report.format_2f),
      ("xprof_bandwidth_per_host_gb_s", report.format_2f),
  )

  def __init__(
      self,
      config: HBMBandwidthParams,
      hardware_spec: system.HardwareSpec,
      mesh: jax.sharding.Mesh | None = None,
      xprof_config: base.XprofConfig | None = None,
  ):
    super().__init__(
        config=config,
        hardware_spec=hardware_spec,
        mesh=mesh,
        xprof_config=xprof_config,
    )
    self.spec: HBMKernelSpec | None = None
    self.scalar: Any | None = None
    self.scope_mesh: jax.sharding.Mesh | None = None
    self.sharding: jax.sharding.NamedSharding | None = None

  def setup(self):
    devices = self.get_devices_to_measure()

    op_type = self.config.op_type
    if op_type in HBM_KERNELS:
      spec = HBM_KERNELS[op_type]
    else:
      raise ValueError(
          f"Unsupported op_type: '{op_type}'. Supported:"
          f" {HBMKernelOp.supported_options_str()}."
      )
    self.spec = spec

    self.scope_mesh = jax.sharding.Mesh(
        np.array(devices), axis_names=(_DEVICE_AXIS,)
    )
    pspec = jax.sharding.PartitionSpec(_DEVICE_AXIS)
    self.sharding = jax.sharding.NamedSharding(self.scope_mesh, pspec)
    replicated_sharding = jax.sharding.NamedSharding(
        self.scope_mesh, jax.sharding.PartitionSpec()
    )

    dtype = dtypes.parse_dtype(self.config.dtype)
    # Any value >= 2 works (integers truncate to >= 2): XLA folds `x * 1` into
    # a copy without the MARKER scope.
    self.scalar = jax.device_put(
        jnp.array(random.uniform(2.0, 5.0), dtype=dtype),
        replicated_sharding,
    )
    scalar = self.scalar
    per_device_shape = (self.config.num_elements,)

    def hbm_op(*args):
      with jax.named_scope(constants.MARKER):
        if spec.num_inputs == 0:
          return spec.kernel_fn(args, scalar, per_device_shape, dtype)
        return spec.kernel_fn(args, scalar)

    sharded_op = jax.shard_map(
        hbm_op,
        mesh=self.scope_mesh,
        in_specs=(pspec,) * spec.num_inputs,
        out_specs=pspec,
        # Forward-only, collective-free per-device kernel: skip VMA checks and
        # implicit `pvary` casts on mesh-invariant scalars (`1.0` in copy and
        # replicated `scalar` in scale/triad, which fail VMA checks when
        # `jax_auto_pcast=False`) and on `write_only`'s invariant `jnp.full`.
        check_vma=False,
    )
    self._jit_fn = jax.jit(sharded_op, out_shardings=self.sharding)

  def get_run_identifier(self) -> str:
    return (
        f"{self.config.op_type}_dim_{self.config.num_elements}"
        f"_scope_{self.config.scope}_idx_{self.config.local_index}"
    )

  def get_devices_to_measure(self) -> list[jax.Device]:
    """Returns the local JAX devices participating in the HBM benchmark."""
    return self.get_devices_for_scope()

  def get_xprof_device_timing_aggregation(self) -> str:
    """Uses slowest-device XProf timing across multi-device scopes by default.

    When `scope=chip` or `scope=host` spans multiple local devices, wall-clock
    timing waits for all devices (`block_until_ready`) and `get_total_bytes()`
    sums bytes across all `get_devices_to_measure()`. Promoting the default
    `measured_device_or_first` mode to `local_slowest_device` ensures XProf
    latency and `xprof_bandwidth_per_{chip,host}_gb_s` reflect the slowest
    device in the scope rather than `devices[0]` alone.

    Returns:
      The effective `XprofDeviceTimingAggregation` value for the active scope.
    """
    if (
        self.xprof_config.device_timing_aggregation
        == constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST
        and len(self.get_devices_to_measure()) > 1
    ):
      return constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE
    return self.xprof_config.device_timing_aggregation

  @property
  def supported_device_timing_aggregations(
      self,
  ) -> frozenset[constants.XprofDeviceTimingAggregation]:
    """Single-device scopes support only `measured_device_or_first`; multi-device scopes support all modes."""
    if self.config.scope == constants.Scope.DEVICE or (
        self.config.scope == constants.Scope.CHIP
        and self.hardware_spec is not None
        and self.hardware_spec.devices_per_chip == 1
    ):
      return frozenset(
          {constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST}
      )
    return super().supported_device_timing_aggregations

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    assert self.spec is not None
    assert self.sharding is not None
    num_devices = self.sharding.num_devices
    num_elements = self.config.num_elements
    dtype = dtypes.parse_dtype(self.config.dtype)
    replicated_sharding = jax.sharding.NamedSharding(
        self.sharding.mesh, jax.sharding.PartitionSpec()
    )

    # Use jit with out_shardings to generate 1D global arrays on device so each
    # device's physical local shard is 1D `(num_elements,)` for compact tiling.
    total_elements = num_devices * num_elements
    generate_data = jax.jit(
        lambda k: utils.random_normal_array(k, (total_elements,), dtype),
        in_shardings=replicated_sharding,
        out_shardings=self.sharding,
    )

    inputs = []
    for i in range(self.spec.num_inputs):
      key = jax.device_put(jax.random.PRNGKey(i), replicated_sharding)
      inputs.append(generate_data(key))
    return tuple(inputs)

  def run_op(self, *args, **kwargs) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args, **kwargs)

  def _get_single_device_bytes(self) -> float:
    assert self.spec is not None
    num_elements = self.config.num_elements
    dtype = dtypes.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    return float(num_elements * itemsize * self.spec.num_arrays)

  def get_total_bytes(self) -> float:
    return len(self.get_devices_to_measure()) * self._get_single_device_bytes()

  def get_arithmetic_intensity(self) -> float:
    assert self.spec is not None
    return (
        float(self.spec.num_flops_per_element * self.config.num_elements)
        / self._get_single_device_bytes()
    )

  def get_workload_metadata(self) -> dict[str, Any]:
    assert self.spec is not None
    return {
        "num_devices": len(self.get_devices_to_measure()),
        "data_size_mib": self._get_single_device_bytes() / (1024 * 1024),
        "op_type": self.spec.name,
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    total_bytes = self.get_total_bytes()

    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      bandwidth_gb_s = float("inf")
    else:
      bandwidth_gb_s = (total_bytes / latency_s) / 1e9

    return self.scope_bandwidth_metrics(prefix, bandwidth_gb_s)
