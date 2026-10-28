"""Collective communication benchmarks."""

import dataclasses
import logging
import math
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
from jax import core
from jax import ffi
from jax.experimental import mesh_utils
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np

_BASE_N = 8
_BASE_K = 128
_REDUCE_SCATTER_K = 256


# 1. Define the Primitive
# pytype: disable=module-attr
Primitive = type(jax.lax.add_p)
# pytype: enable=module-attr
zero_crop_p = Primitive("zero_crop")


# 2. Implement Abstract Evaluation (output shape/dtype is same as input)
def zero_crop_abstract_eval(x):
  return core.ShapedArray(x.shape, x.dtype)


zero_crop_p.def_abstract_eval(zero_crop_abstract_eval)


# 3. Implement the Lowering Rule using jax.ffi
def zero_crop_lowering(ctx, x):
  return ffi.ffi_lowering("ZeroCrop", has_side_effect=True)(ctx, x)


mlir.register_lowering(zero_crop_p, zero_crop_lowering)


# 4. Create a Python Wrapper using jax.ffi.ffi_call
def zero_crop(x):
  if jax.default_backend() == "cpu":
    return x
  return ffi.ffi_call(
      "ZeroCrop",
      result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
      has_side_effect=True,
  )(x)


class ReduceOp(constants.ParamEnum):
  """Reduction operations supported by all-reduce collective benchmark."""

  SUM = "sum"
  MEAN = "mean"
  MAX = "max"
  MIN = "min"


_REDUCE_OP_MAP: dict[ReduceOp, Callable[..., Any]] = {
    ReduceOp.SUM: jax.lax.psum,
    ReduceOp.MEAN: jax.lax.pmean,
    ReduceOp.MAX: jax.lax.pmax,
    ReduceOp.MIN: jax.lax.pmin,
}


class DevicePlacement(constants.ParamEnum):
  """Device placement strategies used when building the collective mesh."""

  MESH_UTILS = "mesh_utils"
  ID_SORTED = "id_sorted"


@dataclasses.dataclass
class CollectivesParams(base.SingleDtypeBenchmarkParams):
  """Topology-neutral parameters shared by all collective benchmarks."""

  matrix_dim: int = dataclasses.field(
      default=1024,
      metadata={
          "min": 1,
          "help": (
              "Dimension size for sharded collective matrices. Actual matrix"
              " size will be (matrix_dim, 8, 128)."
          ),
      },
  )
  seed: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Random seed for tensor initialization."},
  )


@dataclasses.dataclass
class IciCollectivesParams(CollectivesParams):
  """Parameters for collectives over the ICI mesh of a single slice."""

  mesh_shape: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "help": "Logical TPU mesh shape string (e.g. '2x4x4', '2x2x1')."
      },
  )
  device_placement: DevicePlacement = dataclasses.field(
      default=DevicePlacement.MESH_UTILS,
      metadata={
          "help": (
              "Mesh device placement: 'mesh_utils' (topology-aware) or"
              " 'id_sorted' (sorted by device id)."
          )
      },
  )
  sharding_strategy: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          "help": (
              "Device axis sharding strategy string (e.g. '2x2x2', '2x2x1')."
          )
      },
  )


@dataclasses.dataclass
class DcnCollectivesParams(CollectivesParams):
  """Parameters for DCN collective benchmarks."""


# XLA does not implement complex all-reduce or reduce-scatter on TPU.
_NON_COMPLEX_DTYPES = (jnp.bool_, jnp.integer, jnp.floating)


@dataclasses.dataclass
class AllReduceParamsMixin:
  """All-reduce parameters, reusable with any topology params class."""

  DTYPE_CONSTRAINTS = {"dtype": _NON_COMPLEX_DTYPES}

  reduce_op: ReduceOp = dataclasses.field(
      default=ReduceOp.SUM,
      metadata={
          "help": "Reduction operation.",
      },
  )


@dataclasses.dataclass
class AllReduceParams(AllReduceParamsMixin, IciCollectivesParams):
  """Parameters for the ICI all-reduce benchmark."""


@dataclasses.dataclass
class DcnAllReduceParams(AllReduceParamsMixin, DcnCollectivesParams):
  """Parameters for DCN all-reduce collective benchmarks."""


@dataclasses.dataclass
class ReduceScatterParams(IciCollectivesParams):
  """Parameters for the ICI reduce-scatter benchmark."""

  DTYPE_CONSTRAINTS = {"dtype": _NON_COMPLEX_DTYPES}


TCollectiveConfig = TypeVar("TCollectiveConfig", bound=CollectivesParams)
TIciConfig = TypeVar("TIciConfig", bound=IciCollectivesParams)
TDcnConfig = TypeVar("TDcnConfig", bound=DcnCollectivesParams)


class BaseCollectiveBenchmark(
    base.BaseBenchmark[TCollectiveConfig], Generic[TCollectiveConfig]
):
  """Base class for all collective communication benchmarks.

  Holds the topology-neutral logic: input generation, run_op, replica-group
  slicing of the mesh and total bytes. A topology base builds `self.mesh` in
  `setup()` and implements `_get_sharding_axes()`, `get_workload_metadata()`
  and `calculate_throughput_metrics()`.
  """

  Config = CollectivesParams

  def get_run_identifier(self) -> str:
    return f"dim_{self.config.matrix_dim}"

  def _get_sharding_axes(self) -> str | tuple[Any, ...]:
    raise NotImplementedError("Subclasses must implement _get_sharding_axes")

  def _get_num_sharded_devices(
      self, sharding_axes: Optional[str | tuple[Any, ...]] = None
  ) -> int:
    """Returns the number of devices the input is sharded across.

    Args:
      sharding_axes: Mesh axis name or tuple of axis names to shard over. If
        None, computed via `_get_sharding_axes()`.

    Returns:
      The product of the mesh sizes over `sharding_axes`, or 1 if no axis is
      sharded.
    """
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if sharding_axes is None:
      sharding_axes = self._get_sharding_axes()
    if isinstance(sharding_axes, str):
      return self.mesh.shape[sharding_axes]
    return math.prod(self.mesh.shape[axis] for axis in sharding_axes)

  def _setup_jit_fn(self):
    raise NotImplementedError("Subclasses must implement _setup_jit_fn")

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
    # TODO(vvashishth): Verify shapes and sharding match before returning.
    shape = (num_devices, dim, dim)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(sharding_axes, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    dim = self.config.matrix_dim
    dtype = utils.parse_dtype(self.config.dtype)

    sharding_axes = self._get_sharding_axes()
    num_sharded_devices = self._get_num_sharded_devices(sharding_axes)

    shape, sharding = self._get_input_shape_and_sharding(
        num_sharded_devices, dim, sharding_axes
    )

    key = jax.random.PRNGKey(self.config.seed)

    generate_data = jax.jit(
        lambda k: jax.random.normal(k, shape, dtype=dtype),
        out_shardings=sharding,
    )
    data = generate_data(key)

    return (data,)

  def run_op(self, data) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(data)

  def _get_first_replica_group_devices(self) -> list[jax.Device]:
    """Returns the devices of the first replica group, derived from the JAX mesh."""
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    sharding_axes = self._get_sharding_axes()
    sharding_axes_set = (
        {sharding_axes}
        if isinstance(sharding_axes, str)
        else set(sharding_axes)
    )
    indexer = tuple(
        slice(None) if axis in sharding_axes_set else slice(0, 1)
        for axis in self.mesh.axis_names
    )
    return list(self.mesh.devices[indexer].flatten())

  def get_total_bytes(self) -> float:
    return float(self.get_workload_metadata()["data_transferred_bytes"])

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ) -> tuple[float, dict[str, float]]:
    raise NotImplementedError("Subclasses must implement _get_transfer_metrics")


class IciCollectiveBenchmark(
    BaseCollectiveBenchmark[TIciConfig], Generic[TIciConfig]
):
  """Base class for collectives over the ICI mesh of a single slice."""

  Config = IciCollectivesParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("mesh_shape", report.format_str),
      ("device_placement", report.format_str),
      ("sharding_strategy", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_chip_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_chip_gb_s", report.format_2f),
  )

  def __init__(
      self,
      config: TIciConfig,
      hardware_spec: system.HardwareSpec,
      mesh: Optional[jax.sharding.Mesh] = None,
      xprof_config: Optional[base.XprofConfig] = None,
  ):
    super().__init__(
        config=config,
        hardware_spec=hardware_spec,
        mesh=mesh,
        xprof_config=xprof_config,
    )
    self.sharding_strategy = None

  def _build_mesh_devices(self, mesh_shape: list[int]) -> np.ndarray:
    """Returns the mesh device array per `config.device_placement`."""
    if self.config.device_placement == DevicePlacement.ID_SORTED:
      return np.asarray(sorted(jax.devices(), key=lambda d: d.id)).reshape(
          mesh_shape
      )
    return mesh_utils.create_device_mesh(mesh_shape, devices=jax.devices())

  def setup(self):
    mesh_shape_str = self.config.mesh_shape
    if mesh_shape_str is not None:
      try:
        mesh_shape = [int(i) for i in mesh_shape_str.split("x")]
        mesh_devices = self._build_mesh_devices(mesh_shape)
      except (ValueError, RuntimeError) as e:
        print(
            f"Warning: Invalid mesh_shape '{mesh_shape_str}'. Falling back to"
            f" original mesh. Error: {e}"
        )
      else:
        axis_names = tuple(f"d_{i}" for i in range(len(mesh_shape)))
        self.mesh = jax.sharding.Mesh(mesh_devices, axis_names)

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    self.sharding_strategy = self.config.sharding_strategy

    self._setup_jit_fn()

  def _get_sharding_axes(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if self.mesh.axis_names[0] == "device":
      return self.mesh.axis_names[0]

    if self.sharding_strategy is not None:
      try:
        sharding_dims = [int(i) for i in self.sharding_strategy.split("x")]
      except ValueError as e:
        print(
            "Warning: Failed to parse sharding_strategy"
            f" '{self.sharding_strategy}'. Falling back to all mesh axes."
            f" Error: {e}"
        )
      else:
        if len(sharding_dims) != len(self.mesh.shape):
          print(
              "Warning: Failed to parse sharding_strategy"
              f" '{self.sharding_strategy}': length does not match mesh shape"
              f" '{self.mesh.shape}'. Falling back to all mesh axes."
          )
        else:
          return tuple(
              name
              for i, name in enumerate(self.mesh.axis_names)
              if sharding_dims[i] > 1
          )

    return tuple(self.mesh.axis_names)

  def _get_devices_per_chip_in_replica_group(
      self, replica_group_devices: Sequence[jax.Device]
  ) -> int:
    """Returns how many devices of the same chip are in one replica group.

    Args:
      replica_group_devices: Devices of one replica group.

    Returns:
      1 for parallel replica groups (one core per chip per group) and for
      single-device-per-chip hardware; devices_per_chip for non-parallel
      replica groups, where all cores of a chip are in the same group.
    """
    devices_per_chip = self.hardware_spec.devices_per_chip
    # HLO replica_group ids are logical, so decide one-core-per-chip from the
    # mesh devices' core_on_chip.
    if (
        devices_per_chip > 1
        and len(replica_group_devices) > 1
        and len({d.core_on_chip for d in replica_group_devices}) == 1
    ):
      return 1
    return devices_per_chip

  def get_workload_metadata(self) -> dict[str, Any]:
    """Calculate static collective transfer metadata (bytes moved, replica group)."""
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    dim = self.config.matrix_dim
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    first_group_devices = self._get_first_replica_group_devices()
    rank = len(first_group_devices)

    devices_per_chip = self.hardware_spec.devices_per_chip
    devices_per_chip_in_replica_group = (
        self._get_devices_per_chip_in_replica_group(first_group_devices)
    )
    if devices_per_chip > 1 and devices_per_chip_in_replica_group == 1:
      replica_group_type = "parallel"
      participating_ranks = max(rank - 1, 1)
      tf_multiplier = devices_per_chip
    else:
      replica_group_type = "non-parallel"
      participating_ranks = max(rank - devices_per_chip, 1)
      tf_multiplier = 1

    data_transferred_bytes, extra_metrics = self._get_transfer_metrics(
        dim=dim,
        itemsize=itemsize,
        rank=rank,
        participating_ranks=participating_ranks,
        tf_multiplier=tf_multiplier,
    )
    return {
        "data_transferred_bytes": data_transferred_bytes,
        "replica_group_type": replica_group_type,
        "replica_group_rank": rank,
        **extra_metrics,
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    metadata = self.get_workload_metadata()
    latency_s = latency_ms / 1000.0
    if metadata["replica_group_rank"] > 1:
      if latency_s == 0:
        bandwidth_gb_s = float("inf")
      else:
        bandwidth_gb_s = metadata["data_transferred_bytes"] / (latency_s * 1e9)
    else:
      bandwidth_gb_s = 0.0

    return {
        f"{prefix}_bandwidth_per_chip_gb_s": bandwidth_gb_s,
    }


class DcnCollectiveBenchmark(
    BaseCollectiveBenchmark[TDcnConfig], Generic[TDcnConfig]
):
  """Base class for collectives over the DCN mesh across slices."""

  Config = DcnCollectivesParams
  derive_chip_bandwidth: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_host_gbps", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_host_gbps", report.format_2f),
  )

  def setup(self):
    devices = jax.devices()
    device_key = lambda d: (*d.coords, int(d.core_on_chip))
    slice_ids = sorted({getattr(d, "slice_index", 0) for d in devices})
    if len(slice_ids) < 2:
      raise ValueError(
          f"{type(self).__name__} requires at least 2 slices for DCN "
          f"collectives, but found {len(slice_ids)} slice(s) "
          f"(slice_ids={slice_ids})."
      )
    slices = [
        sorted(
            [d for d in devices if getattr(d, "slice_index", 0) == sid],
            key=device_key,
        )
        for sid in slice_ids
    ]
    slice_counts = {sid: len(s) for sid, s in zip(slice_ids, slices)}
    if len(set(slice_counts.values())) > 1:
      raise ValueError(
          "All slices must have the same number of devices, but got "
          f"per-slice device counts: {slice_counts}."
      )
    self.mesh = jax.sharding.Mesh(
        np.array(slices, dtype=object), ("dcn", "ici")
    )
    self._setup_jit_fn()

  def _get_sharding_axes(self):
    return "dcn"

  def get_workload_metadata(self) -> dict[str, Any]:
    """Calculate static DCN collective transfer metadata."""
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    dim = self.config.matrix_dim
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    rank = len(self._get_first_replica_group_devices())
    participating_ranks = max(rank - 1, 1)

    data_transferred_bytes, extra_metrics = self._get_transfer_metrics(
        dim=dim,
        itemsize=itemsize,
        rank=rank,
        participating_ranks=participating_ranks,
        tf_multiplier=1,
    )
    return {
        "data_transferred_bytes": data_transferred_bytes,
        "replica_group_rank": rank,
        **extra_metrics,
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    latency_s = latency_ms / 1000.0
    per_device_gbps = self.get_total_bytes() * 8.0 / (latency_s * 1e9)
    return {
        f"{prefix}_bandwidth_per_host_gbps": (
            per_device_gbps * jax.local_device_count()
        ),
    }


class AllReduceOpMixin(
    BaseCollectiveBenchmark[TCollectiveConfig], Generic[TCollectiveConfig]
):
  """All-reduce op logic shared by any topology base.

  Provides reduce_op validation, the run identifier, the replicated input,
  the jitted all-reduce and its transfer bytes. List it before the topology
  base, whose config must include `AllReduceParamsMixin`. The topology base
  builds `self.mesh` in `setup()`, returns its collective axes from
  `_get_sharding_axes()` and passes rank, participating ranks and the
  transfer multiplier to `_get_transfer_metrics()` from its own
  `get_workload_metadata()`.

  `_setup_jit_fn()` always wraps the op in the MARKER named scope and
  `zero_crop`, with no hook to change either. Until such a hook exists, a
  non-ICI topology must override `_setup_jit_fn()` itself; the rest of this
  mixin is reusable as is.
  """

  def _get_reduce_op(self) -> ReduceOp:
    config = self.config
    if not isinstance(config, AllReduceParamsMixin):
      raise TypeError(
          f"{type(self).__name__} requires a config that subclasses"
          f" AllReduceParamsMixin, got {type(config).__name__}."
      )
    return config.reduce_op

  def setup(self):
    op = self._get_reduce_op()
    if op not in _REDUCE_OP_MAP:
      raise ValueError(
          f"Invalid reduce_op '{op}'. "
          f"Must be one of {ReduceOp.supported_options_str()}"
      )
    super().setup()

  def get_run_identifier(self) -> str:
    dim = self.config.matrix_dim
    op = self._get_reduce_op()
    return f"dim_{dim}_op_{op}"

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
    shape = (dim, _BASE_N, _BASE_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _setup_jit_fn(self):
    sharding_axes = self._get_sharding_axes()
    op_fn = _REDUCE_OP_MAP[self._get_reduce_op()]

    @jax.jit
    def all_reduce_sharded(x):
      def f(a):
        with jax.named_scope(constants.MARKER):
          # Insert the custom call to prevent result from being a live out buffer
          return zero_crop(op_fn(a, axis_name=sharding_axes))

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=jax.sharding.PartitionSpec(None, None, None),
          out_specs=jax.sharding.PartitionSpec(None, None, None),
          check_vma=False,
      )(x)

    self._jit_fn = all_reduce_sharded

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = (
        2
        * local_size_bytes
        * (participating_ranks / max(rank, 1))
        * tf_multiplier
    )
    return data_transferred, {"shard_size_mib": local_size_bytes / (1024 * 1024)}


@registry.benchmark_registry.register("all_reduce")
class AllReduceBenchmark(
    AllReduceOpMixin[AllReduceParams], IciCollectiveBenchmark[AllReduceParams]
):
  """Benchmarks latency and bandwidth of all-reduce collective ops across devices."""

  Config = AllReduceParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("reduce_op", report.format_str),
      ("mesh_shape", report.format_str),
      ("device_placement", report.format_str),
      ("sharding_strategy", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_chip_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_chip_gb_s", report.format_2f),
  )


@registry.benchmark_registry.register("dcn_all_reduce")
class DcnAllReduceBenchmark(
    AllReduceOpMixin[DcnAllReduceParams],
    DcnCollectiveBenchmark[DcnAllReduceParams],
):
  """Benchmarks multi-slice DCN AllReduce latency and bandwidth."""

  Config = DcnAllReduceParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("reduce_op", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_host_gbps", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_host_gbps", report.format_2f),
  )

  def _setup_jit_fn(self):
    sharding_axes = self._get_sharding_axes()
    op_fn = _REDUCE_OP_MAP[self._get_reduce_op()]

    @jax.jit
    def all_reduce_sharded(x):
      def f(a):
        return op_fn(a, axis_name=sharding_axes)

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=jax.sharding.PartitionSpec(None, None, None),
          out_specs=jax.sharding.PartitionSpec(None, None, None),
          check_vma=False,
      )(x)

    self._jit_fn = all_reduce_sharded


@registry.benchmark_registry.register("all_gather")
class AllGatherBenchmark(IciCollectiveBenchmark[IciCollectivesParams]):
  """Benchmarks the latency and bandwidth of jax.lax.all_gather across devices."""

  def _setup_jit_fn(self):
    sharding_axes = self._get_sharding_axes()

    @jax.jit
    def all_gather_sharded(x):
      def f(a):
        with jax.named_scope(constants.MARKER):
          return jax.lax.all_gather(
              a,
              axis_name=sharding_axes,
              tiled=True,
          )

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=jax.sharding.PartitionSpec(None, None, None),
          out_specs=jax.sharding.PartitionSpec(None, None, None),
          check_vma=False,
      )(x)

    self._jit_fn = all_gather_sharded

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ):
    shape = (dim, _BASE_N, _BASE_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = local_size_bytes * participating_ranks * tf_multiplier
    return data_transferred, {"shard_size_mib": local_size_bytes / (1024 * 1024)}


def all_to_all_peak_bandwidth_per_chip_gb_s(
    replica_group_rank: int,
    devices_per_chip_in_replica_group: int,
    replica_group_chip_coords: Sequence[Sequence[int]],
    topology_dims: Sequence[int],
    link_bw_per_direction_gb_s: float,
    wraparound_rule: system.WraparoundRule,
) -> float:
  """Returns the theoretical peak all-to-all bus bandwidth per chip (GB/s).

  The all-to-all is bounded by the bisection bandwidth of the replica group.
  Bus-bandwidth view:

      peak_bw_per_chip = 4 * B * M * P / R^2 * k

  Derivation (S = input bytes per device, D = devices per chip):
    1. Bytes crossing the bisection, per direction: R/2 devices on one side
       each send one S/R chunk to each of the R/2 devices on the other side:
       (R/2) * (R/2) * (S/R) = R*S/4.
    2. Bisection bandwidth: M links * B per link = M*B.
    3. Roofline runtime: D/k replica groups run concurrently on the same chips
       (2 for parallel replica groups on 2-device chips, otherwise 1) and
       share the M links: T = (D/k) * (R*S/4) / (M*B).
    4. Bus bytes per device: S * P/R.
    5. Peak bus bandwidth per device: (S*P/R) / T = 4*B*M*P / R^2 * k/D.
    6. Peak per chip: x D, the same per-chip scaling as the measured bytes in
       `AllToAllBenchmark._get_transfer_metrics`. Unlike all-gather/all-reduce,
       every (source -> destination) chunk of an all-to-all is distinct, so
       each core of a chip sends its own chunks over ICI. D cancels, leaving
       4*B*M*P / R^2 * k, so parallel and non-parallel replica groups on the
       same chips share the same per-chip peak.

  Args:
    replica_group_rank: R, devices in one replica group.
    devices_per_chip_in_replica_group: k, devices of the same chip in that
      replica group (1 for parallel replica groups, devices_per_chip
      otherwise).
    replica_group_chip_coords: Chip coordinates of the replica group's devices
      (devices on the same chip share coordinates).
    topology_dims: Chip-level extent of the full ICI topology on each axis.
    link_bw_per_direction_gb_s: B, theoretical per-link ICI BW per direction.
    wraparound_rule: Which topology axes have wrap-around links.

  Returns:
    The peak bus bandwidth per chip in decimal GB/s.

  Raises:
    ValueError: If the bisection model does not apply to this replica group.
  """
  # --- Replica group shape + guards --------------------------------------
  chips = {tuple(c) for c in replica_group_chip_coords}
  replica_group_dims = platform.bounding_box_dims(list(chips))
  if len(chips) <= 1:
    raise ValueError("replica group spans a single chip (die-to-die only)")
  # The model assumes every chip contributes the same k devices; asymmetric
  # groups (e.g. both cores of one chip but one core of the others) do not
  # fit it.
  if replica_group_rank != len(chips) * devices_per_chip_in_replica_group:
    raise ValueError(
        f"replica group has {replica_group_rank} devices, not {len(chips)}"
        f" chips x {devices_per_chip_in_replica_group} devices per chip"
    )
  # The group is a solid box iff its chips fill every position of their
  # bounding box; otherwise it is scattered (e.g. chips at x=0 and x=2 only),
  # its traffic hops through chips outside the group, and the box model below
  # does not apply.
  is_contiguous_box = len(chips) == math.prod(replica_group_dims)
  if not is_contiguous_box:
    raise ValueError(
        f"replica group's {len(chips)} chips do not form a contiguous"
        f" {'x'.join(map(str, replica_group_dims))} box"
    )
  # TODO(simonleesyuan): Validate on hardware for replica groups spanning a
  # subset of the ICI topology (e.g. sharding_strategy 4x4x1 on a 4x4x4
  # topology). So far only replica groups spanning the full topology (every
  # checked-in config) are validated.

  # --- R, k, P, B -------------------------------------------------------
  rank = replica_group_rank  # R
  k = devices_per_chip_in_replica_group  # k
  participating_ranks = rank - k  # P: peers reached over ICI (not die-to-die)
  link_bw = link_bw_per_direction_gb_s  # B

  # --- M: unidirectional links crossing the minimum bisection ------------
  # Which topology axes have physical wrap-around links (hardware rule).
  if wraparound_rule == system.WraparoundRule.ALL_DIMS_GE_4:
    all_wrap = len(topology_dims) >= 3 and all(d >= 4 for d in topology_dims)
    topology_wraps = [all_wrap] * len(topology_dims)
  elif wraparound_rule == system.WraparoundRule.PER_DIM_GE_16:
    topology_wraps = [d >= 16 for d in topology_dims]
  else:
    topology_wraps = [False] * len(topology_dims)
  # Try halving the replica group across each axis it spans; the bisection is
  # the cut crossed by the fewest links.
  links_per_cut = []
  for cut_axis, extent in enumerate(replica_group_dims):
    if extent <= 1:
      continue
    # Links the cut crosses: the product of the other dims, e.g. cutting a
    # 2x4x4 replica group across a 4-axis crosses 2*4 = 8 links.
    cross_section_links = math.prod(
        d for axis, d in enumerate(replica_group_dims) if axis != cut_axis
    )
    # A wrap-around link is a second path across the cut (a ring cut in half
    # is cut in two places). It only counts if the replica group spans the
    # whole axis; otherwise the wrap-around link leads to chips outside it.
    spans_full_axis = extent == topology_dims[cut_axis]
    wrap_factor = 2 if (topology_wraps[cut_axis] and spans_full_axis) else 1
    links_per_cut.append(wrap_factor * cross_section_links)
  bisection_link_multiplier = min(links_per_cut)  # M

  # --- Peak -------------------------------------------------------------
  # 4 = 2 * 2: 1/2 of the senders are on one side of the cut, and 1/2 of
  # each sender's destinations are on the other side (step 1 above).
  bisection_divisor = 4
  # Steps 5-6: the per-device peak (x k/D) times D per chip; D cancels.
  return (
      bisection_divisor
      * link_bw
      * bisection_link_multiplier
      * participating_ranks
      / rank**2
      * k
  )


@registry.benchmark_registry.register("all_to_all")
class AllToAllBenchmark(IciCollectiveBenchmark[IciCollectivesParams]):
  """Benchmarks the latency and bandwidth of jax.lax.all_to_all across devices.

  The theoretical ICI peak (and its derivation) is in
  `all_to_all_peak_bandwidth_per_chip_gb_s`.
  """

  roofline_mode: constants.RooflineMode = constants.RooflineMode.NETWORK_ICI
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      *IciCollectiveBenchmark.REPORT_SCHEMA,
      ("peak_ici_bandwidth_per_chip_gb_s", report.format_2f),
      ("wall_clock_ici_roofline_efficiency_pct", report.format_2f),
      ("xprof_ici_roofline_efficiency_pct", report.format_2f),
  )

  def _get_topology_devices(self) -> Sequence[jax.Device]:
    """Returns every device in the ICI topology."""
    return jax.devices()

  def get_peak_ici_bandwidth_per_chip(self) -> Optional[float]:
    """Returns the all-to-all ICI peak, or None (with a warning) if N/A.

    The peak is not modeled (and `*_ici_roofline_efficiency_pct` is omitted)
    for:
      - Missing ICI spec.
      - Non-TPU devices without chip coordinates.
      - Multi-slice runs (cross-slice traffic uses DCN).
      - Replica groups on a single chip (die-to-die only).
      - Replica groups whose chips do not form a contiguous box.
    """
    ici = self.hardware_spec.ici
    # The replica group whose bandwidth is measured.
    replica_group_devices = self._get_first_replica_group_devices()
    # Every device in the job; the replica group is a subset of it.
    topology_devices = self._get_topology_devices()
    try:
      if ici is None:  # `HardwareSpec.ici` is Optional; shipped specs set it.
        raise ValueError(f"no ICI spec for {self.hardware_spec.name}")
      if any(getattr(d, "coords", None) is None for d in topology_devices):
        raise ValueError("devices have no physical chip coordinates (non-TPU)")
      if len({getattr(d, "slice_index", 0) for d in topology_devices}) > 1:
        raise ValueError(
            "multi-slice run; cross-slice traffic uses DCN, not ICI"
        )
      return all_to_all_peak_bandwidth_per_chip_gb_s(
          replica_group_rank=len(replica_group_devices),
          devices_per_chip_in_replica_group=(
              self._get_devices_per_chip_in_replica_group(
                  replica_group_devices
              )
          ),
          replica_group_chip_coords=[d.coords for d in replica_group_devices],
          topology_dims=platform.bounding_box_dims(
              [d.coords for d in topology_devices]
          ),
          link_bw_per_direction_gb_s=ici.link_bw_per_direction_gb_s,
          wraparound_rule=ici.wraparound_rule,
      )
    except ValueError as e:
      logging.warning(
          "Skipping all_to_all *_ici_roofline_efficiency_pct: %s.", e
      )
      return None

  def _setup_jit_fn(self):
    sharding_axes = self._get_sharding_axes()

    @jax.jit
    def all_to_all_sharded(x):
      def f(a):
        with jax.named_scope(constants.MARKER):
          return jax.lax.all_to_all(
              a,
              axis_name=sharding_axes,
              split_axis=0,
              concat_axis=0,
              tiled=True,
          )

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=jax.sharding.PartitionSpec(None, None, None),
          out_specs=jax.sharding.PartitionSpec(None, None, None),
          check_vma=False,
      )(x)

    self._jit_fn = all_to_all_sharded

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ):
    shape = (dim, _BASE_N, _BASE_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    # In All-to-All, every (source_rank -> dest_rank) chunk is distinct, so all
    # devices_per_chip cores on a chip independently transmit their own outbound
    # inter-chip data in both parallel and non-parallel replica group modes.
    data_transferred = (
        local_size_bytes
        * (participating_ranks / max(rank, 1))
        * self.hardware_spec.devices_per_chip
    )
    return data_transferred, {
        "shard_size_mib": local_size_bytes / (1024 * 1024)
    }


@registry.benchmark_registry.register("reduce_scatter", is_experimental=True)
class ReduceScatterBenchmark(IciCollectiveBenchmark[ReduceScatterParams]):
  """[EXPERIMENTAL] Benchmarks the latency and bandwidth of jax.lax.psum_scatter across devices."""

  Config = ReduceScatterParams

  def _setup_jit_fn(self):
    sharding_axes = self._get_sharding_axes()

    @jax.jit
    def reduce_scatter_sharded(x):
      def f(a):
        with jax.named_scope(constants.MARKER):
          return jax.lax.psum_scatter(
              a,
              axis_name=sharding_axes,
              tiled=True,
          )

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=jax.sharding.PartitionSpec(None, None, None),
          out_specs=jax.sharding.PartitionSpec(sharding_axes, None, None),
          check_vma=False,
      )(x)

    self._jit_fn = reduce_scatter_sharded

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ):
    shape = (num_devices, dim, _REDUCE_SCATTER_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    chunk_size_bytes = dim * _REDUCE_SCATTER_K * itemsize
    data_transferred = (
        chunk_size_bytes * (participating_ranks / max(rank, 1)) * tf_multiplier
    )
    return data_transferred, {"shard_size_mib": chunk_size_bytes / (1024 * 1024)}
