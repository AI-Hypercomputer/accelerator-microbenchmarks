"""Collective communication benchmarks."""

import dataclasses
from typing import Any, Optional
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
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


@dataclasses.dataclass
class CollectivesParams(base.BaseBenchmarkParams):
  mesh_shape: Optional[str] = None
  sharding_strategy: Optional[str] = None
  matrix_dim: int = 1024
  dtype: str = "bfloat16"
  seed: int = 0


class BaseCollectiveBenchmark(base.BaseBenchmark[CollectivesParams]):
  Config = CollectivesParams
  """Base class for all collective communication benchmarks."""

  def __init__(
      self, config: CollectivesParams, mesh: Optional[jax.sharding.Mesh] = None
  ):
    super().__init__(config, mesh)
    self.sharding_strategy = None

  def setup(self):
    mesh_shape_str = self.config.mesh_shape
    if mesh_shape_str is not None:
      try:

        mesh_shape = [int(i) for i in mesh_shape_str.split("x")]
        axis_names = tuple(f"d_{i}" for i in range(len(mesh_shape)))
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

    avg_latency_s = metrics["avg_ms"] / 1000.0

    data_transferred_bytes, extra_metrics = self._get_transfer_metrics(
        dim=dim, itemsize=itemsize, num_devices=sharding_size
    )

    if sharding_size > 1:
      bandwidth_gb_s = data_transferred_bytes / (avg_latency_s * 1e9)
    else:
      bandwidth_gb_s = 0.0

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics.update(extra_metrics)
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

    bytes_moved, _ = self._get_transfer_metrics(
        dim=dim, itemsize=itemsize, num_devices=sharding_size
    )
    return bytes_moved

  def get_arithmetic_intensity(self) -> float:
    return 0.0

  def _get_transfer_metrics(
      self, dim: int, itemsize: int, num_devices: int
  ) -> tuple[float, dict[str, float]]:
    raise NotImplementedError("Subclasses must implement _get_transfer_metrics")


@registry.benchmark_registry.register("all_reduce_sum")
class AllReduceSumBenchmark(BaseCollectiveBenchmark):
  """Benchmarks the latency and bandwidth of jax.lax.psum across devices."""

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

    @jax.jit
    def psum_sharded(x):
      def f(a):
        with jax.named_scope(constants.MARKER):
          # Insert the custom call to prevent result from being a live out buffer
          return zero_crop(jax.lax.psum(a, axis_name=sharding_axes))

      return jax.shard_map(
          f,
          mesh=self.mesh,
          in_specs=jax.sharding.PartitionSpec(None, None, None),
          out_specs=jax.sharding.PartitionSpec(None, None, None),
          check_vma=False,
      )(x)

    self._jit_fn = psum_sharded

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = local_size_bytes * 2 * (num_devices - 1) / num_devices
    return data_transferred, {"shard_size_mb": local_size_bytes / 1e6}


@registry.benchmark_registry.register("all_gather")
class AllGatherBenchmark(BaseCollectiveBenchmark):
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

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    data_transferred = local_size_bytes * (num_devices - 1)
    return data_transferred, {"shard_size_mb": local_size_bytes / 1e6}


@registry.benchmark_registry.register("all_to_all")
class AllToAllBenchmark(BaseCollectiveBenchmark):
  """Benchmarks the latency and bandwidth of jax.lax.all_to_all across devices."""

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

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    local_size_bytes = dim * _BASE_N * _BASE_K * itemsize
    chunk_size_bytes = local_size_bytes / num_devices
    data_transferred = chunk_size_bytes * (num_devices - 1)
    return data_transferred, {"local_size_mb": local_size_bytes / 1e6}


@registry.benchmark_registry.register("reduce_scatter")
class ReduceScatterBenchmark(BaseCollectiveBenchmark):
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

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    chunk_size_bytes = dim * _REDUCE_SCATTER_K * itemsize
    data_transferred = chunk_size_bytes * (num_devices - 1)
    return data_transferred, {"shard_size_mb": chunk_size_bytes / 1e6}
