"""A script to run the microbenchmarks in Jax over DCN and ICI collectives."""

# pylint: disable=g-importing-member
from functools import partial
from typing import Any, Dict, Tuple

from benchmark_utils import simple_timeit
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
# pylint: disable=g-importing-member


def create_mesh(
    dcn_size: int, ici_size: int
) -> tuple[Mesh, list[int], list[int]]:
  """Creates a hybrid mesh with the given DCN and ICI sizes."""
  dcn_parallelism = [dcn_size, 1]
  ici_parallelism = [1, ici_size]

  total_devices = jax.device_count()
  if total_devices != (dcn_size * ici_size):
    raise ValueError(
        f"Need {dcn_size * ici_size} devices, but found {total_devices}"
    )
  if dcn_size > 1:
    mesh_devices = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism, dcn_parallelism, devices=jax.devices()
    )
    mesh = Mesh(mesh_devices, ("dcn", "ici"))
  else:
    mesh_devices = mesh_utils.create_device_mesh(
        [ici_size], devices=jax.devices()
    )
    mesh = Mesh(mesh_devices, "ici")
  return mesh, dcn_parallelism, ici_parallelism


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Helper function to build the metrics and metadata for the benchmark."""
  metrics_keys = ["ici_average_time_ms", "dcn_average_time_ms"]
  metadata = {
      key: value
      for key, value in params
      if value is not None and key not in metrics_keys
  }
  metadata["dtype"] = metadata["dtype"].dtype.itemsize
  metrics = {key: value for key, value in params if key in metrics_keys}
  return metadata, metrics


def psum_benchmark(
    matrix_dim: int, dtype: jnp.dtype, dcn_size: int, ici_size: int
) -> Dict[str, Any]:
  """Benchmarks the psum collective operation.

  Args:
    matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
      matrix_dim).
    dtype: The data type of the matrix.
    dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
      benchmark is run.
    ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
      is run. The ICI and DCN

  Returns:
    The measured time for the DCN and ICI benchmarks.
  """
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)

  dcn_average_time_ms = ici_average_time_ms = None
  # DCN benchmark
  if dcn_size > 1:

    @partial(shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P(None))
    def psum_dcn_op(x):
      return jax.lax.psum(x, "dcn")

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(psum_dcn_op)
    dcn_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_dcn_op"
    )

  # ICI benchmark
  if ici_size > 1:
    @partial(
        shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, None)
    )
    def psum_ici_op(x):
      return jax.lax.psum(x, "ici")

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, None))
    )
    jitted_op = jax.jit(psum_ici_op)
    ici_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_ici_op"
    )
  return {
      "dcn_average_time_ms": dcn_average_time_ms,
      "ici_average_time_ms": ici_average_time_ms,
  }


def psum_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms: float,
    dcn_average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the psum benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metadata, metrics = get_metrics_helper(params)

  matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
  dcn_bandwidth_gbyte_s = ici_bandwidth_gbyte_s = None
  # Calculate metrics for DCN benchmark
  if dcn_size > 1 and dcn_average_time_ms is not None:
    # bandwidth is claculated as psum can be done via reduce_scatter +
    # all_gather so bandwidth is the sum of the two (formulas below)
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        * 2
        / dcn_size
        / dcn_size
        / (dcn_average_time_ms / 1e3)
    )
    dcn_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"psum_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )
  # Calculate metrics for ICI benchmark
  if ici_size > 1 and ici_average_time_ms is not None:
    # bandwidth is claculated as psum can be done via reduce_scatter +
    # all_gather so bandwidth is the sum of the two (formulas below)
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        * 2
        / ici_size
        / (ici_average_time_ms / 1e3)
    )
    ici_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"psum_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )
  # Gather the metrics to report.
  metrics.update({
      "ici_bandwidth_gbyte_s": ici_bandwidth_gbyte_s,
      "dcn_bandwidth_gbyte_s": dcn_bandwidth_gbyte_s,
  })
  return metadata, metrics


def psum_scatter_benchmark(
    matrix_dim: int, dtype: jnp.dtype, dcn_size: int, ici_size: int
) -> Dict[str, Any]:
  """Benchmarks the psum_scatter collective operation.

  Args:
    matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
      matrix_dim).
    dtype: The data type of the matrix.
    dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
      benchmark is run.
    ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
      is run. The ICI and DCN

  Returns:
    The measured time for the DCN and ICI benchmarks.
  """
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  dcn_average_time_ms = ici_average_time_ms = None
  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def psum_scatter_dcn_op(x):
      return jax.lax.psum_scatter(x, "dcn", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(psum_scatter_dcn_op)
    dcn_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_scatter_dcn_op"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, "ici")
    )
    def psum_scatter_ici_op(x):
      return jax.lax.psum_scatter(x, "ici", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, None))
    )
    jitted_op = jax.jit(psum_scatter_ici_op)
    ici_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="psum_scatter_ici_op"
    )

  return {
      "dcn_average_time_ms": dcn_average_time_ms,
      "ici_average_time_ms": ici_average_time_ms,
  }


def psum_scatter_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms: float,
    dcn_average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the psum_scatter benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metadata, metrics = get_metrics_helper(params)

  matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
  dcn_bandwidth_gbyte_s = ici_bandwidth_gbyte_s = None
  # Calculate metrics for DCN benchmark
  if dcn_size > 1 and dcn_average_time_ms is not None:

    # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
    # to use (dcn_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        / dcn_size
        / dcn_size
        / (dcn_average_time_ms / 1e3)
    )
    dcn_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"psum_scatter_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # Calculate metrics for ICI benchmark
  if ici_size > 1 and ici_average_time_ms is not None:
    # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
    # to use (ici_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        / ici_size
        / (ici_average_time_ms / 1e3)
    )
    ici_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"psum_scatter_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )
  # Gather the metrics to report.
  metrics.update({
      "ici_bandwidth_gbyte_s": ici_bandwidth_gbyte_s,
      "dcn_bandwidth_gbyte_s": dcn_bandwidth_gbyte_s,
  })
  metrics = {key: value for key, value in metrics.items() if value is not None}
  return metadata, metrics


def all_gather_benchmark(
    matrix_dim: int, dtype: jnp.dtype, dcn_size: int, ici_size: int
) -> Dict[str, Any]:
  """Benchmarks the all_gather collective operation.

  Args:
    matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
      matrix_dim).
    dtype: The data type of the matrix.
    dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
      benchmark is run.
    ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
      is run. The ICI and DCN

  Returns:
    The measured time for the DCN and ICI benchmarks.
  """
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  dcn_average_time_ms = ici_average_time_ms = None

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def all_gather_dcn_op(x):
      return jax.lax.all_gather(x, "dcn", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(all_gather_dcn_op)
    dcn_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_gather_dcn_op"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, None),
        check_rep=False,
    )
    def all_gather_ici_op(x):
      return jax.lax.all_gather(x, "ici", tiled=True)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, None))
    )
    jitted_op = jax.jit(all_gather_ici_op)
    ici_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_gather_ici_op"
    )

  return {
      "dcn_average_time_ms": dcn_average_time_ms,
      "ici_average_time_ms": ici_average_time_ms,
  }


def all_gather_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms: float,
    dcn_average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the all_gather benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metadata, metrics = get_metrics_helper(params)

  matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
  dcn_bandwidth_gbyte_s = ici_bandwidth_gbyte_s = None
  # Calculate metrics for DCN benchmark
  if dcn_size > 1 and dcn_average_time_ms is not None:

    # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
    # to use (dcn_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        / dcn_size
        / (dcn_average_time_ms / 1e3)
    )
    dcn_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"all_gather_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # Calculate metrics for ICI benchmark
  if ici_size > 1 and ici_average_time_ms is not None:
    # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
    # to use (ici_size - 1) steps in a ring algorithm
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        / (ici_average_time_ms / 1e3)
    )
    ici_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"all_gather_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )
  # Gather the metrics to report.
  metrics.update({
      "ici_bandwidth_gbyte_s": ici_bandwidth_gbyte_s,
      "dcn_bandwidth_gbyte_s": dcn_bandwidth_gbyte_s,
  })
  metrics = {key: value for key, value in metrics.items() if value is not None}
  return metadata, metrics


def ppermute_benchmark(
    matrix_dim: int, dtype: jnp.dtype, dcn_size: int, ici_size: int
) -> Dict[str, Any]:
  """Benchmarks the ppermute collective operation.

  Args:
    matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
      matrix_dim).
    dtype: The data type of the matrix.
    dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
      benchmark is run.
    ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
      is run. The ICI and DCN

  Returns:
    The measured time for the DCN and ICI benchmarks.
  """
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  dcn_average_time_ms = ici_average_time_ms = None

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def ppermute_dcn_op(x):
      perm = [(i, (i + 1) % dcn_size) for i in range(dcn_size)]
      return jax.lax.ppermute(x, "dcn", perm)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(ppermute_dcn_op)
    dcn_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="ppermute_dcn_op"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, "ici")
    )
    def ppermute_ici_op(x):
      perm = [(i, (i + 1) % ici_size) for i in range(ici_size)]
      return jax.lax.ppermute(x, "ici", perm)

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, None))
    )
    jitted_op = jax.jit(ppermute_ici_op)
    ici_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="ppermute_ici_op"
    )

  return {
      "dcn_average_time_ms": dcn_average_time_ms,
      "ici_average_time_ms": ici_average_time_ms,
  }


def ppermute_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms: float,
    dcn_average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the ppermute benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metadata, metrics = get_metrics_helper(params)

  matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
  dcn_bandwidth_gbyte_s = ici_bandwidth_gbyte_s = None
  # Calculate metrics for DCN benchmark
  if dcn_size > 1 and dcn_average_time_ms is not None:

    # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
    # to use 1 step
    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte / dcn_size / (dcn_average_time_ms / 1e3)
    )
    dcn_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"ppermute_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # Calculate metrics for ICI benchmark
  if ici_size > 1 and ici_average_time_ms is not None:
    # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
    # to use 1 step
    achieved_bandwidth_gbyte_s = matrix_size_gbyte / (ici_average_time_ms / 1e3)
    ici_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"ppermute_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  metrics.update({
      "ici_bandwidth_gbyte_s": ici_bandwidth_gbyte_s,
      "dcn_bandwidth_gbyte_s": dcn_bandwidth_gbyte_s,
  })
  return metadata, metrics


def all_to_all_benchmark(
    matrix_dim: int, dtype: jnp.dtype, dcn_size: int, ici_size: int
) -> Dict[str, Any]:
  """Benchmarks the all_to_all collective operation.

  Args:
    matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
      matrix_dim).
    dtype: The data type of the matrix.
    dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
      benchmark is run.
    ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
      is run. The ICI and DCN

  Returns:
    The measured time for the DCN and ICI benchmarks.
  """
  mesh, _, _ = create_mesh(dcn_size, ici_size)
  matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
  dcn_average_time_ms = ici_average_time_ms = None

  # DCN benchmark
  if dcn_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P("dcn", None), out_specs=P("dcn", None)
    )
    def all_to_all_dcn_op(x):
      return jax.lax.all_to_all(
          x, "dcn", split_axis=0, concat_axis=0, tiled=True
      )

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P("dcn", None))
    )
    jitted_op = jax.jit(all_to_all_dcn_op)
    ici_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_to_all_dcn_op"
    )

  # ICI benchmark
  if ici_size > 1:

    @partial(
        shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, None),
        check_rep=False,
    )
    def all_to_all_ici_op(x):
      return jax.lax.all_to_all(
          x, "ici", split_axis=0, concat_axis=0, tiled=True
      )

    sharded_matrix = jax.device_put(
        matrix, jax.sharding.NamedSharding(mesh, P(None, None))
    )
    jitted_op = jax.jit(all_to_all_ici_op)
    ici_average_time_ms = simple_timeit(
        jitted_op, sharded_matrix, task="all_to_all_ici_op"
    )

  return {
      "dcn_average_time_ms": dcn_average_time_ms,
      "ici_average_time_ms": ici_average_time_ms,
  }


def all_to_all_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms: float,
    dcn_average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the all_to_all benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metadata, metrics = get_metrics_helper(params)

  matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
  dcn_bandwidth_gbyte_s = ici_bandwidth_gbyte_s = None
  # Calculate metrics for DCN benchmark
  if dcn_size > 1 and dcn_average_time_ms is not None:

    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (dcn_size - 1)
        / dcn_size
        / dcn_size
        / (dcn_average_time_ms / 1e3)
    )
    dcn_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"all_to_all_dcn: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )

  # Calculate metrics for ICI benchmark
  if ici_size > 1 and ici_average_time_ms is not None:

    achieved_bandwidth_gbyte_s = (
        matrix_size_gbyte
        * (ici_size - 1)
        / ici_size
        / (ici_average_time_ms / 1e3)
    )
    ici_bandwidth_gbyte_s = achieved_bandwidth_gbyte_s
    print(
        f"all_to_all_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {achieved_bandwidth_gbyte_s=}"
    )
  # Gather the metrics to report.
  metrics.update({
      "ici_bandwidth_gbyte_s": ici_bandwidth_gbyte_s,
      "dcn_bandwidth_gbyte_s": dcn_bandwidth_gbyte_s,
  })
  metrics = {key: value for key, value in metrics.items() if value is not None}
  return metadata, metrics
