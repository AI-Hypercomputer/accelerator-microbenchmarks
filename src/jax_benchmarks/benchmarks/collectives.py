"""Collective communication benchmarks."""

from typing import Any, Dict, List, Optional, Tuple
import jax
from jax import core
from jax import ffi
from jax import shard_map
from jax._src.core import Primitive
from jax.interpreters import mlir
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
from ..core.base import BaseBenchmark
from ..core.registry import registry

# 1. Define the Primitive
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
  return ffi.ffi_call(
      "ZeroCrop",
      result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
      has_side_effect=True,
  )(x)


class BaseCollectiveBenchmark(BaseBenchmark):
  """Base class for all collective communication benchmarks."""

  def setup(self, **params):
    mesh_shape_str = params.get("mesh_shape", None)
    if mesh_shape_str is not None:
      try:
        from jax.experimental import mesh_utils
        from jax.sharding import Mesh

        mesh_shape = [int(i) for i in mesh_shape_str.split("x")]
        axis_names = tuple(f"d_{i}" for i in range(len(mesh_shape)))
        mesh_devices = mesh_utils.create_device_mesh(
            mesh_shape, devices=jax.devices()
        )
        self.mesh = Mesh(mesh_devices, axis_names)
      except Exception as e:
        print(
            f"Warning: Invalid mesh_shape '{mesh_shape_str}'. Falling back to"
            f" original mesh. Error: {e}"
        )

    if self.mesh is None:
      raise ValueError("Mesh not initialized.")

    self._setup_jit_fn(**params)

  def _get_sharding_axes(self):
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    if self.mesh.axis_names[0] == "device":
      return self.mesh.axis_names[0]
    else:
      return tuple(self.mesh.axis_names)

  def _setup_jit_fn(self, **params):
    raise NotImplementedError("Subclasses must implement _setup_jit_fn")

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ) -> Tuple[tuple, NamedSharding]:
    shape = (num_devices, dim, dim)
    sharding = NamedSharding(self.mesh, P(sharding_axes, None, None))
    return shape, sharding

  def generate_inputs(self, **params) -> Tuple[jnp.ndarray]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    dim = params.get("matrix_dim", 1024)
    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str)

    num_devices = self.mesh.devices.size
    sharding_axes = self._get_sharding_axes()

    shape, sharding = self._get_input_shape_and_sharding(
        num_devices, dim, sharding_axes
    )

    key = jax.random.PRNGKey(params.get("seed", 0))

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

  def calculate_metrics(
      self, times_ms: List[float], **params
  ) -> Dict[str, Any]:
    if self.mesh is None:
      raise ValueError("Mesh not initialized.")
    metrics = super().calculate_metrics(times_ms, **params)

    dim = params.get("matrix_dim", 1024)
    dtype_str = params.get("dtype", "bfloat16")
    dtype = getattr(jnp, dtype_str)
    itemsize = jnp.dtype(dtype).itemsize

    num_devices = self.mesh.devices.size
    avg_latency_s = metrics["avg_ms"] / 1000.0

    data_transferred_bytes, extra_metrics = self._get_transfer_metrics(
        dim=dim, itemsize=itemsize, num_devices=num_devices
    )

    if num_devices > 1:
      bandwidth_gb_s = data_transferred_bytes / (avg_latency_s * 1e9)
    else:
      bandwidth_gb_s = 0.0

    metrics["bandwidth_gb_s"] = bandwidth_gb_s
    metrics.update(extra_metrics)
    return metrics

  def _get_transfer_metrics(
      self, dim: int, itemsize: int, num_devices: int
  ) -> Tuple[float, Dict[str, float]]:
    raise NotImplementedError("Subclasses must implement _get_transfer_metrics")


@registry.register("all_reduce_sum")
class AllReduceSumBenchmark(BaseCollectiveBenchmark):
  """Benchmarks the latency and bandwidth of jax.lax.psum across devices."""

  def _setup_jit_fn(self, **params):
    sharding_axes = self._get_sharding_axes()
    from ..core.constants import MARKER

    @jax.jit
    def psum_sharded(x):
      def f(a):
        # Insert the custom call to prevent result from being a live out buffer
        return zero_crop(jax.lax.psum(a, axis_name=sharding_axes))

      with jax.named_scope(MARKER):
        return shard_map(
            f,
            mesh=self.mesh,
            in_specs=P(sharding_axes, None, None),
            out_specs=P(sharding_axes, None, None),
            check_vma=False,
        )(x)

    self._jit_fn = psum_sharded

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    shard_size_bytes = dim * dim * itemsize
    data_transferred = shard_size_bytes * 2 * (num_devices - 1) / num_devices
    return data_transferred, {"shard_size_mb": shard_size_bytes / 1e6}


@registry.register("all_gather")
class AllGatherBenchmark(BaseCollectiveBenchmark):
  """Benchmarks the latency and bandwidth of jax.lax.all_gather across devices."""

  def _setup_jit_fn(self, **params):
    sharding_axes = self._get_sharding_axes()
    from ..core.constants import MARKER

    @jax.jit
    def all_gather_sharded(x):
      with jax.named_scope(MARKER):
        return shard_map(
            lambda a: jax.lax.all_gather(a, axis_name=sharding_axes, tiled=True),
            mesh=self.mesh,
            in_specs=P(None, None, None),
            out_specs=P(None, None, None),
            check_vma=False,
        )(x)

    self._jit_fn = all_gather_sharded

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ):
    shape = (1, dim, dim)
    sharding = NamedSharding(self.mesh, P(None, None, None))
    return shape, sharding

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    shard_size_bytes = dim * dim * itemsize
    data_transferred = shard_size_bytes * (num_devices - 1)
    return data_transferred, {"shard_size_mb": shard_size_bytes / 1e6}


@registry.register("all_to_all")
class AllToAllBenchmark(BaseCollectiveBenchmark):
  """Benchmarks the latency and bandwidth of jax.lax.all_to_all across devices."""

  def _setup_jit_fn(self, **params):
    sharding_axes = self._get_sharding_axes()
    from ..core.constants import MARKER

    @jax.jit
    def all_to_all_sharded(x):
      with jax.named_scope(MARKER):
        return shard_map(
            lambda a: jax.lax.all_to_all(
                a,
                axis_name=sharding_axes,
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ),
            mesh=self.mesh,
            in_specs=P(None, None, None),
            out_specs=P(None, None, None),
            check_vma=False,
        )(x)

    self._jit_fn = all_to_all_sharded

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ):
    shape = (num_devices, dim, dim)
    sharding = NamedSharding(self.mesh, P(None, None, None))
    return shape, sharding

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    local_size_bytes = num_devices * dim * dim * itemsize
    chunk_size_bytes = dim * dim * itemsize
    data_transferred = chunk_size_bytes * (num_devices - 1)
    return data_transferred, {"local_size_mb": local_size_bytes / 1e6}


@registry.register("reduce_scatter")
class ReduceScatterBenchmark(BaseCollectiveBenchmark):
  """Benchmarks the latency and bandwidth of jax.lax.psum_scatter across devices."""

  def _setup_jit_fn(self, **params):
    sharding_axes = self._get_sharding_axes()
    from ..core.constants import MARKER

    @jax.jit
    def reduce_scatter_sharded(x):
      with jax.named_scope(MARKER):
        return shard_map(
            lambda a: jax.lax.psum_scatter(
                a, axis_name=sharding_axes, tiled=True
            ),
            mesh=self.mesh,
            in_specs=P(None, None, None),
            out_specs=P(sharding_axes, None, None),
            check_vma=False,
        )(x)

    self._jit_fn = reduce_scatter_sharded

  def _get_input_shape_and_sharding(
      self, num_devices: int, dim: int, sharding_axes
  ):
    shape = (num_devices, dim, dim)
    sharding = NamedSharding(self.mesh, P(None, None, None))
    return shape, sharding

  def _get_transfer_metrics(self, dim: int, itemsize: int, num_devices: int):
    chunk_size_bytes = dim * dim * itemsize
    data_transferred = chunk_size_bytes * (num_devices - 1)
    return data_transferred, {"shard_size_mb": chunk_size_bytes / 1e6}
