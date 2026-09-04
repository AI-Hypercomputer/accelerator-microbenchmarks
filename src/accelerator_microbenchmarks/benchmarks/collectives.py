"""Collective communication benchmarks."""

import dataclasses
import glob
import os
import re
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
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


_REDUCE_OP_MAP = {
    "sum": jax.lax.psum,
    "mean": jax.lax.pmean,
    "max": jax.lax.pmax,
    "min": jax.lax.pmin,
}


@dataclasses.dataclass
class CollectivesParams(base.BaseBenchmarkParams):
  mesh_shape: Optional[str] = None
  mesh_axes: Optional[str] = None
  sharding_strategy: Optional[str] = None
  matrix_dim: int = 1024
  dtype: str = "bfloat16"
  seed: int = 0
  xla_dump_dir: Optional[str] = "/tmp/xla_dump"
  input_layout: str = "default"
  ring_factor: Optional[float] = None


@dataclasses.dataclass
class AllReduceParams(CollectivesParams):
  reduce_op: str = dataclasses.field(
      default="sum",
      metadata={
          "help": "Reduction operation: sum, mean, max, min",
      },
  )


TCollectiveConfig = TypeVar("TCollectiveConfig", bound=CollectivesParams)


class BaseCollectiveBenchmark(
    base.BaseBenchmark[TCollectiveConfig], Generic[TCollectiveConfig]
):
  """Base class for all collective communication benchmarks."""

  Config = CollectivesParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("mesh_shape", report.format_str),
      ("sharding_strategy", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("bandwidth_gb_s", report.format_2f),
      ("p50_ms", report.format_4f),
      ("xprof_p50_ms", report.format_4f),
  )

  def __init__(
      self,
      config: TCollectiveConfig,
      hardware_spec: system.HardwareSpec,
      mesh: Optional[jax.sharding.Mesh] = None,
  ):
    super().__init__(config=config, hardware_spec=hardware_spec, mesh=mesh)
    self.sharding_strategy = None

  def setup(self):
    mesh_shape_str = self.config.mesh_shape
    mesh_axes_str = self.config.mesh_axes
    if mesh_shape_str is not None:
      try:
        mesh_shape = [int(i) for i in mesh_shape_str.split("x")]
        if mesh_axes_str is not None:
          axis_names = tuple(mesh_axes_str.split(","))
        else:
          axis_names = tuple(f"d_{i}" for i in range(len(mesh_shape)))

        if "dcn" in axis_names and len(mesh_shape) == 2:
          self.mesh, _ = utils.build_multislice_mesh(
              mesh_shape[0], mesh_shape[1], axis_names=axis_names
          )
        else:
          mesh_devices = mesh_utils.create_device_mesh(
              mesh_shape, devices=jax.devices()
          )
          self.mesh = jax.sharding.Mesh(mesh_devices, axis_names)
      except (ValueError, RuntimeError) as e:
        print(
            f"Warning: Invalid mesh_shape '{mesh_shape_str}'. Falling back to"
            f" original mesh. Error: {e}"
        )

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    self.sharding_strategy = self.config.sharding_strategy

    self._setup_jit_fn()

  def get_run_identifier(self) -> str:
    dim = self.config.matrix_dim
    if dim is not None:
      return f"dim_{dim}"
    return ""

  def _get_sharding_axes(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if self.mesh.axis_names[0] == "device":
      return self.mesh.axis_names[0]

    if self.sharding_strategy is not None:
      try:
        sharding_dims = [int(i) for i in self.sharding_strategy.split("x")]
        if len(sharding_dims) != len(self.mesh.shape):
          raise ValueError(
              f"sharding_strategy '{self.sharding_strategy}' length does not"
              f" match mesh shape '{self.mesh.shape}'"
          )
        sharding_axes = tuple(
            name
            for i, name in enumerate(self.mesh.axis_names)
            if sharding_dims[i] > 1
        )
        return sharding_axes
      except Exception as e:
        print(
            "Warning: Failed to parse sharding_strategy"
            f" '{self.sharding_strategy}'. Falling back to all mesh axes."
            f" Error: {e}"
        )

    return tuple(self.mesh.axis_names)

  def _setup_jit_fn(self):
    raise NotImplementedError("Subclasses must implement _setup_jit_fn")

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
    # TODO(vvashishth): Verify shapes and sharding match before returning.
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if getattr(self.config, "input_layout", "default") == "dcn_1d":
      shape = (dim, dim)
      sharding = jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec("dcn", None)  # pyrefly: ignore[bad-argument-type]
      )
      return shape, sharding
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
    if isinstance(sharding_axes, str):
      sharding_size = self.mesh.shape[sharding_axes]
    else:
      sharding_size = 1
      for axis in sharding_axes:
        sharding_size *= self.mesh.shape[axis]

    shape, sharding = self._get_input_shape_and_sharding(
        sharding_size, dim, sharding_axes
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

  def _extract_first_replica_group_from_hlo_dump(self) -> list[int]:
    """Reads disk-based HLO dump files and extracts the first replica group."""
    search_dirs = []
    if self.config.xla_dump_dir:
      search_dirs.append(self.config.xla_dump_dir)

    xla_flags = os.environ.get("XLA_FLAGS", "")
    match = re.search(r"--xla_dump_to=([^ ]+)", xla_flags)
    if match:
      search_dirs.append(match.group(1))

    for dump_dir in search_dirs:
      if not os.path.exists(dump_dir):
        continue
      files = glob.glob(
          os.path.join(dump_dir, "*after_optimizations*.txt")
      ) + glob.glob(os.path.join(dump_dir, "*.txt"))
      files.sort(key=os.path.getmtime, reverse=True)
      for fpath in files:
        if os.path.isfile(fpath):
          try:
            with open(fpath, "r") as f:
              content = f.read()
            rg_match = re.search(
                r"replica_groups=({{[0-9,]+(?:},{[0-9,]+)*}})",
                content,
                re.DOTALL,
            )
            if rg_match:
              content_rg = rg_match.group(1)[2:-2]
              first_group_str = content_rg.split("},{")[0]
              return [int(x) for x in first_group_str.split(",")]
          except Exception:
            pass

    # Fallback for CPU unit testing when no disk HLO is dumped
    if self.mesh:
      sharding_axes = self._get_sharding_axes()
      if isinstance(sharding_axes, str):
        sharding_size = self.mesh.shape[sharding_axes]
      else:
        sharding_size = 1
        for axis in sharding_axes:
          sharding_size *= self.mesh.shape[axis]
      return list(range(sharding_size))

    raise ValueError(
        "Could not find or parse replica_groups from disk HLO dump files in"
        f" search directories: {search_dirs}"
    )

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    metrics = super().calculate_metrics(times_ms)

    dim = self.config.matrix_dim
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    sharding_axes = self._get_sharding_axes()
    if isinstance(sharding_axes, str):
      sharding_size = self.mesh.shape[sharding_axes]
    else:
      sharding_size = 1
      for axis in sharding_axes:
        sharding_size *= self.mesh.shape[axis]

    try:
      first_replica_group = self._extract_first_replica_group_from_hlo_dump()
      rank = len(first_replica_group)

      if first_replica_group and all(i % 2 == 0 for i in first_replica_group):
        replica_group_type = "parallel"
        participating_ranks = max(rank - 1, 1)
        tf_multiplier = 2
      else:
        replica_group_type = "non-parallel"
        participating_ranks = max(rank - 2, 1)
        tf_multiplier = 1
    except Exception as e:
      replica_group_type = "non-parallel"
      rank = sharding_size
      participating_ranks = max(rank - 1, 1)
      tf_multiplier = 1
      print(
          "Warning: Failed to extract replica group from HLO dump. Falling"
          f" back to non-parallel replica group. Error: {e}"
      )

    avg_latency_s = metrics["avg_ms"] / 1000.0

    data_transferred_bytes, extra_metrics = self._get_transfer_metrics(
        dim=dim,
        itemsize=itemsize,
        num_devices=sharding_size,
        rank=rank,
        participating_ranks=participating_ranks,
        tf_multiplier=tf_multiplier,
    )

    if sharding_size > 1:
      bandwidth_gb_s = data_transferred_bytes / (avg_latency_s * 1e9)
    else:
      bandwidth_gb_s = 0.0

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics["replica_group_type"] = replica_group_type
    metrics["replica_group_rank"] = rank
    metrics.update(extra_metrics)

    if getattr(self.config, "input_layout", "default") == "dcn_1d":
      shard_bytes = extra_metrics.get(
          "algorithm_bytes", extra_metrics.get("shard_size_bytes", 0.0)
      )
      ring_factor = extra_metrics.get("ring_factor", 1.0)
      participants = extra_metrics.get("participants_per_slice", 1)
      if avg_latency_s > 0 and shard_bytes > 0:
        per_device_algo_gbps = (shard_bytes * 8.0) / (avg_latency_s * 1e9)
        host_algo_gbps = per_device_algo_gbps * participants
        host_bus_gbps = host_algo_gbps * ring_factor
        metrics["per_device_algo_gbps"] = per_device_algo_gbps
        metrics["host_algo_gbps"] = host_algo_gbps
        metrics["host_bus_gbps"] = host_bus_gbps
        # Dual-NIC line rate on TPU7x / v6e: 400 Gbps TX + 400 Gbps RX = 800 Gbps duplex
        metrics["duplex_utilization_pct"] = (host_bus_gbps / 800.0) * 100.0
    return metrics

  def get_total_bytes(self) -> float:
    dim = self.config.matrix_dim
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    if self.mesh:
      sharding_axes = self._get_sharding_axes()
      if isinstance(sharding_axes, str):
        sharding_size = self.mesh.shape[sharding_axes]
      else:
        sharding_size = 1
        for axis in sharding_axes:
          sharding_size *= self.mesh.shape[axis]
    else:
      sharding_size = 1

    try:
      first_replica_group = self._extract_first_replica_group_from_hlo_dump()
      rank = len(first_replica_group)
      if first_replica_group and all(i % 2 == 0 for i in first_replica_group):
        participating_ranks = max(rank - 1, 1)
        tf_multiplier = 2
      else:
        participating_ranks = max(rank - 2, 1)
        tf_multiplier = 1
    except Exception:
      rank = sharding_size
      participating_ranks = max(rank - 1, 1)
      tf_multiplier = 1

    bytes_moved, _ = self._get_transfer_metrics(
        dim=dim,
        itemsize=itemsize,
        num_devices=sharding_size,
        rank=rank,
        participating_ranks=participating_ranks,
        tf_multiplier=tf_multiplier,
    )
    return bytes_moved

  def get_arithmetic_intensity(self) -> float:
    return 0.0

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      num_devices: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ) -> tuple[float, dict[str, float]]:
    raise NotImplementedError("Subclasses must implement _get_transfer_metrics")


@registry.benchmark_registry.register("all_reduce")
class AllReduceBenchmark(BaseCollectiveBenchmark[AllReduceParams]):
  """Benchmarks latency and bandwidth of all-reduce collective ops across devices."""

  Config = AllReduceParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("reduce_op", report.format_str),
      ("mesh_shape", report.format_str),
      ("sharding_strategy", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("bandwidth_gb_s", report.format_2f),
      ("p50_ms", report.format_4f),
      ("xprof_p50_ms", report.format_4f),
  )

  def setup(self):
    op = self.config.reduce_op.lower()
    if op not in _REDUCE_OP_MAP:
      raise ValueError(
          f"Invalid reduce_op '{self.config.reduce_op}'. "
          f"Must be one of {list(_REDUCE_OP_MAP.keys())}"
      )
    super().setup()

  def get_run_identifier(self) -> str:
    dim = self.config.matrix_dim
    op = self.config.reduce_op.lower()
    return f"dim_{dim}_op_{op}"

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if getattr(self.config, "input_layout", "default") == "dcn_1d":
      shape = (dim, dim)
      sharding = jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec("dcn", None)  # pyrefly: ignore[bad-argument-type]
      )
      return shape, sharding
    shape = (dim, _BASE_N, _BASE_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _setup_jit_fn(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    sharding_axes = self._get_sharding_axes()
    op_fn = _REDUCE_OP_MAP[self.config.reduce_op.lower()]
    is_dcn_1d = getattr(self.config, "input_layout", "default") == "dcn_1d"

    if is_dcn_1d:
      target_axis = "dcn" if "dcn" in self.mesh.axis_names else sharding_axes

      @jax.jit
      def all_reduce_dcn(x):
        def f(a):
          with jax.named_scope(constants.MARKER):
            return zero_crop(op_fn(a, axis_name=target_axis))

        return jax.shard_map(
            f,
            mesh=self.mesh,
            in_specs=jax.sharding.PartitionSpec("dcn", None),
            out_specs=jax.sharding.PartitionSpec("dcn", None),
            check_vma=False,
        )(x)

      self._jit_fn = all_reduce_dcn
    else:

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
      num_devices: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    if getattr(self.config, "input_layout", "default") == "dcn_1d":
      num_slices = 2
      participants_per_slice = max(1, num_devices // 2)
      if self.mesh is not None:
        axis_names = tuple(self.mesh.axis_names)
        devices_shape = self.mesh.devices.shape
        if "dcn" in axis_names:
          num_slices = devices_shape[axis_names.index("dcn")]
        if "ici" in axis_names:
          participants_per_slice = devices_shape[axis_names.index("ici")]
        else:
          participants_per_slice = max(1, num_devices // num_slices)
      shard_bytes = dim * dim * itemsize / num_slices
      ring_factor = (
          self.config.ring_factor
          if self.config.ring_factor is not None
          else (2.0 * (num_slices - 1) / num_slices)
      )
      data_transferred = shard_bytes * ring_factor * participants_per_slice
      return data_transferred, {
          "shard_size_mib": shard_bytes / (1024 * 1024),
          "algorithm_bytes": shard_bytes,
          "ring_factor": ring_factor,
          "participants_per_slice": participants_per_slice,
      }
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = (
        2
        * local_size_bytes
        * (participating_ranks / max(rank, 1))
        * tf_multiplier
    )
    return data_transferred, {"shard_size_mib": local_size_bytes / (1024 * 1024)}


@registry.benchmark_registry.register("all_gather")
class AllGatherBenchmark(BaseCollectiveBenchmark[CollectivesParams]):
  """Benchmarks the latency and bandwidth of jax.lax.all_gather across devices."""

  def match_xprof_op_fallback(self, event: dict[str, Any]) -> bool:
    # copybara:strip_begin(b/553842170: temporary fallback until XLA compiler bug is fixed)
    # TODO(b/553842170): Remove this fallback once the underlying compiler issue
    # (MARKER failing to attach to operation in 2x4x4 all gather) is resolved.
    # copybara:strip_end
    args = event.get("args", {})
    hlo_category = args.get("hlo_category", "")
    offload_type = args.get("offload_type", "")
    return (
        hlo_category == "async-done"
        and offload_type == "OFFLOAD_COLLECTIVE"
    )

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
      num_devices: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = local_size_bytes * participating_ranks * tf_multiplier
    return data_transferred, {"shard_size_mib": local_size_bytes / (1024 * 1024)}


@registry.benchmark_registry.register("all_to_all")
class AllToAllBenchmark(BaseCollectiveBenchmark[CollectivesParams]):
  """Benchmarks the latency and bandwidth of jax.lax.all_to_all across devices."""

  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("mesh_shape", report.format_str),
      ("sharding_strategy", report.format_str),
      ("matrix_dim", report.format_str),
      ("local_size_mib", report.format_2f),
      ("bandwidth_gb_s", report.format_2f),
      ("p50_ms", report.format_4f),
      ("xprof_p50_ms", report.format_4f),
  )

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
    shape = (dim * num_devices, _BASE_N, _BASE_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      num_devices: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = (
        local_size_bytes * (participating_ranks / max(rank, 1)) * tf_multiplier
    )
    return data_transferred, {"local_size_mib": local_size_bytes / (1024 * 1024)}


@registry.benchmark_registry.register("reduce_scatter")
class ReduceScatterBenchmark(BaseCollectiveBenchmark[CollectivesParams]):
  """Benchmarks the latency and bandwidth of jax.lax.psum_scatter across devices."""

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
      num_devices: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    chunk_size_bytes = dim * _REDUCE_SCATTER_K * itemsize
    data_transferred = (
        chunk_size_bytes * (participating_ranks / max(rank, 1)) * tf_multiplier
    )
    return data_transferred, {"shard_size_mib": chunk_size_bytes / (1024 * 1024)}


@dataclasses.dataclass
class PPermuteParams(CollectivesParams):
  direction: str = dataclasses.field(
      default="uni",
      metadata={
          "help": (
              "Permutation direction: uni (unidirectional [(0, 1)]) or bidi"
              " (bidirectional [(0, 1), (1, 0)])"
          ),
      },
  )


@registry.benchmark_registry.register("ppermute")
@registry.benchmark_registry.register("ppermute_uni")
@registry.benchmark_registry.register("ppermute_bidi")
class PPermuteBenchmark(BaseCollectiveBenchmark[PPermuteParams]):
  """Benchmarks latency and bandwidth of ppermute peer-to-peer collective ops across devices."""

  Config = PPermuteParams
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("direction", report.format_str),
      ("mesh_shape", report.format_str),
      ("sharding_strategy", report.format_str),
      ("matrix_dim", report.format_str),
      ("shard_size_mib", report.format_2f),
      ("bandwidth_gb_s", report.format_2f),
      ("p50_ms", report.format_4f),
      ("xprof_p50_ms", report.format_4f),
  )

  def setup(self):
    super().setup()
    direction = self.config.direction.lower()
    if direction not in ("uni", "bidi"):
      raise ValueError(
          f"Invalid direction '{self.config.direction}'. Must be 'uni' or"
          " 'bidi'."
      )

  def get_run_identifier(self) -> str:
    dim = self.config.matrix_dim
    direction = self.config.direction.lower()
    return f"dim_{dim}_dir_{direction}"

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if getattr(self.config, "input_layout", "default") == "dcn_1d":
      shape = (dim, dim)
      sharding = jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec("dcn", None)  # pyrefly: ignore[bad-argument-type]
      )
      return shape, sharding
    shape = (dim, _BASE_N, _BASE_K)
    sharding = jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(None, None, None)  # pyrefly: ignore[bad-argument-type]
    )
    return shape, sharding

  def _setup_jit_fn(self):
    direction = self.config.direction.lower()
    perm = [(0, 1), (1, 0)] if direction == "bidi" else [(0, 1)]
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    target_axis = (
        "dcn" if "dcn" in self.mesh.axis_names else self._get_sharding_axes()
    )
    if isinstance(target_axis, (tuple, list)):
      target_axis = target_axis[0]
    is_dcn_1d = getattr(self.config, "input_layout", "default") == "dcn_1d"

    in_specs = (
        jax.sharding.PartitionSpec("dcn", None)
        if is_dcn_1d
        else jax.sharding.PartitionSpec(None, None, None)
    )

    @jax.jit
    def ppermute_sharded(x):
      def f(a):
        with jax.named_scope(constants.MARKER):
          return zero_crop(
              jax.lax.ppermute(a, axis_name=target_axis, perm=perm)
          )

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=in_specs,
          out_specs=in_specs,
          check_vma=False,
      )(x)

    self._jit_fn = ppermute_sharded

  def _get_transfer_metrics(
      self,
      dim: int,
      itemsize: int,
      num_devices: int,
      rank: int = 1,
      participating_ranks: int = 1,
      tf_multiplier: int = 1,
  ):
    if getattr(self.config, "input_layout", "default") == "dcn_1d":
      num_slices = 2
      participants_per_slice = max(1, num_devices // 2)
      if self.mesh is not None:
        axis_names = tuple(self.mesh.axis_names)
        devices_shape = self.mesh.devices.shape
        if "dcn" in axis_names:
          num_slices = devices_shape[axis_names.index("dcn")]
        if "ici" in axis_names:
          participants_per_slice = devices_shape[axis_names.index("ici")]
        else:
          participants_per_slice = max(1, num_devices // num_slices)
      shard_bytes = dim * dim * itemsize / num_slices
      mult = 2 if self.config.direction.lower() == "bidi" else 1
      data_transferred = shard_bytes * mult * participants_per_slice
      return data_transferred, {
          "shard_size_mib": shard_bytes / (1024 * 1024),
          "algorithm_bytes": shard_bytes,
          "ring_factor": float(mult),
          "participants_per_slice": participants_per_slice,
      }
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    mult = 2 if self.config.direction.lower() == "bidi" else 1
    data_transferred = local_size_bytes * mult
    return data_transferred, {
        "shard_size_mib": local_size_bytes / (1024 * 1024)
    }
