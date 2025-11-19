"""A script to run the microbenchmarks in Jax over ICI collectives."""

from functools import partial
import json
import os
from typing import Any, Dict

import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from benchmark_utils import (
    get_lhs_named_shading,
    get_out_sharding,
    MetricsStatistics,
    multiple_iteration_timeit_from_trace,
    ShardingStrategy,
    simple_timeit,
)
from common import MARKER


BASE_SHAPE= [1,8,128]
SEED = 0
SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING


def create_mesh(
    ici_size: int, mesh_shape: str
) -> Mesh:
    """Creates a mesh with the given ICI size."""
    devices_needed = ici_size
    devices = jax.devices()

    if len(devices) < devices_needed:
        raise ValueError(f"Need {devices_needed} devices, but found {len(devices)}")
    devices = devices[:devices_needed]
    mesh_shape = mesh_shape.split("x")
    mesh_shape = [int(i) for i in mesh_shape]

    shape = mesh_shape if mesh_shape else (ici_size,)

    axis_names = [f"d_{i}" for i in range(len(shape))]

    first_device = devices[0]
    device_kind = first_device.device_kind
    print("Device kind: ", device_kind)
    mesh_devices = mesh_utils.create_device_mesh(shape, devices=jax.devices())
    mesh = Mesh(mesh_devices, axis_names)
    return mesh


def get_metrics_helper(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_keys = ["ici_average_time_ms"]
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_keys
    }
    metadata["dtype"] = metadata["dtype"].dtype.itemsize
    return metadata

def psum_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str = None,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
  """Benchmarks the psum collective operation.

  Args:
    matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
      matrix_dim).
    dtype: The data type of the matrix.
    ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
      is run.
    mesh_shape: The shape of the mesh.

  Returns:
    The measured time for the ICI benchmark.
  """
  libtpu_init_args = [
      "--xla_jf_debug_level=3",
      "--xla_sc_disable_megacore_partitioning=true",
      "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
      "--xla_tpu_enable_all_reduce_offload_tracing=true",
      "--xla_tpu_enable_all_reduce_scatter_fusion=false",
      "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
      "--xla_tpu_pad_operations_input_tiles=true",
      "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0",
      "--xla_tpu_use_tc_device_shape_on_sc=true",
  ]
  os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
  mesh = create_mesh(ici_size, mesh_shape)
  key = jax.random.key(SEED)
  lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
  out_sharding = get_out_sharding(SHARDING_STRATEGY)
  op_dimension_tuple = op_dimension.split("x")
  op_dimension_tuple = tuple(int(dim) for dim in op_dimension_tuple)
  sharding_axis = tuple(
      name
      for i, name in enumerate(mesh.axis_names)
      if op_dimension_tuple[i] > 1
  )

  def f(x):
    with jax.named_scope(MARKER):
      return jax.lax.psum(x, sharding_axis)

  jit_sharded_f = jax.jit(
      shard_map(
          f,
          mesh,
          in_specs=lhs_sharding.spec,
          out_specs=out_sharding,
          check_rep=False,
      )
  )
  m = matrix_dim
  n = BASE_SHAPE[1]
  k = BASE_SHAPE[2]

  def data_generator():
    """Creates new random data on host and puts it on device."""
    nonlocal key  # Use and update the outer 'key'

    matrix = jnp.ones((m, n, k), dtype=dtype)
    return (matrix,)

  print("Running psum benchmark", num_runs, matrix_dim)
  time_ms_list = multiple_iteration_timeit_from_trace(
      jit_sharded_f,
      data_generator,
      matrix_dim=f"{m}x{n}x{k}",
      tries=num_runs,
      task="gemm_throttling",
      trace_dir=trace_dir,
  )
  return {
      "ici_average_time_ms_list": time_ms_list,
  }


def psum_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    input_num_elements = matrix_dim * BASE_SHAPE[1] * BASE_SHAPE[2]
    metadata.update({
        "input_num_elements": input_num_elements,
        "matrix_shape": json.dumps(f"({matrix_dim}, {BASE_SHAPE[1]}, {BASE_SHAPE[2]})"),
    })
    return metadata, metrics


def psum_scatter_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str = None,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the psum_scatter collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run.
      mesh_shape: The shape of the mesh.

    Returns:
      The measured time for the ICI benchmark.
    """
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
        "--xla_tpu_enable_reduce_scatter_offload_tracing=true",
        "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true",
        "--xla_tpu_use_tc_device_shape_on_sc=true",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)

    op_dimension_tuple = op_dimension.split("x")
    op_dimension_tuple = tuple(int(dim) for dim in op_dimension_tuple)
    sharding_axis = tuple(
        name for i, name in enumerate(mesh.axis_names) if op_dimension_tuple[i] > 1
    )

    def f(x):
        with jax.named_scope(MARKER):
            return jax.lax.psum_scatter(x, sharding_axis, tiled=True)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh=mesh,
            in_specs=P(None, None, None),
            out_specs=P(sharding_axis, None, None),
            check_rep=False,
        )
    )

    m = matrix_dim
    n = BASE_SHAPE[1]
    k = BASE_SHAPE[2]

    def data_generator():
        """Creates new random data on host and puts it on device."""
        matrix = jnp.ones((m, n, k), dtype=dtype)
        return (matrix,)

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="psum_scatter_ici_op",
        trace_dir=trace_dir,
    )
    print("Running psum_scatter benchmark", num_runs, matrix_dim)
    return {
        "ici_average_time_ms_list": time_ms_list,
    }


def psum_scatter_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum_scatter benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    input_num_elements = matrix_dim * BASE_SHAPE[1] * BASE_SHAPE[2]
    metadata.update({
        "input_num_elements": input_num_elements,
        "matrix_shape": json.dumps(f"({matrix_dim}, {BASE_SHAPE[1]}, {BASE_SHAPE[2]}"),
    })

    metrics = {key: value for key, value in metrics.items() if value is not None}

    return metadata, metrics


def all_gather_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str = None,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the all_gather collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run.
      mesh_shape: The shape of the mesh.
      op_dimension: The dimension of the operation.
      num_runs: The number of runs to perform.
      trace_dir: The directory to save the trace to.

    Returns:
      The measured time for the ICI benchmark.
    """
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
        "--xla_tpu_enable_all_gather_offload_tracing=true",
        "--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true",
        "--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true",
        "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true",
        "--xla_tpu_use_single_sparse_core_for_all_gather_offload=true",
        "--xla_tpu_use_tc_device_shape_on_sc=true",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)

    op_dimension_tuple = op_dimension.split("x")
    op_dimension_tuple = tuple(int(dim) for dim in op_dimension_tuple)
    sharding_axis = tuple(
        name for i, name in enumerate(mesh.axis_names) if op_dimension_tuple[i] > 1
    )

    def f(x):
        with jax.named_scope(MARKER):
            return jax.lax.all_gather(x, sharding_axis, tiled=True)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh=mesh,
            in_specs=P(None, None, None),
            out_specs=P(None, None, None),
            check_rep=False,
        )
    )
    m = matrix_dim
    n = BASE_SHAPE[1]
    k = BASE_SHAPE[2]

    def data_generator():
        """Creates new random data on host and puts it on device."""
        matrix = jnp.ones((m, n, k), dtype=dtype)
        return (matrix,)

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="all_gather_ici_op",
        trace_dir=trace_dir,
    )
    print("Running all_gather benchmark", num_runs, matrix_dim)
    return {
        "ici_average_time_ms_list": time_ms_list,
    }


def all_gather_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_gather benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    input_num_elements = matrix_dim * BASE_SHAPE[1] * BASE_SHAPE[2]
    metadata.update({
        "input_num_elements": input_num_elements,
        "matrix_shape": json.dumps(f"({matrix_dim}, 8, 128)"),
    })
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def ppermute_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the ppermute collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run.
      mesh_shape: The shape of the mesh.

    Returns:
      The measured time for the ICI benchmark.
    """
    mesh = create_mesh(ici_size, mesh_shape)
    matrix = jnp.ones((matrix_dim, BASE_SHAPE[1], BASE_SHAPE[2]), dtype=dtype)
    ici_average_time_ms_list = None

    # ICI benchmark
    if ici_size > 1:

        @partial(shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, None))
        def f(x):
            perm = [(i, (i + 1) % ici_size) for i in range(ici_size)]
            return jax.lax.ppermute(x, "ici", perm)

        sharded_matrix = jax.device_put(
            matrix, jax.sharding.NamedSharding(mesh, P(None, None))
        )
        jitted_op = jax.jit(f)
        ici_average_time_ms_list = simple_timeit(
            jitted_op,
            sharded_matrix,
            matrix_dim=matrix_dim,
            tries=num_runs,
            task="ppermute_ici_op",
            trace_dir=trace_dir,
        )

    return {
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def ppermute_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the ppermute benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.itemsize / 1e9

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_average_time_ms_list is not None:
        # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
        # to use 1 step
        ici_bandwidth_gbyte_s_list = [
            matrix_size_gbyte / (ici_average_time_ms / 1e3)
            for ici_average_time_ms in ici_average_time_ms_list
        ]
        ici_bandwidth_gbyte_s_statistics = MetricsStatistics(
            metrics_list=ici_bandwidth_gbyte_s_list,
            metrics_name="ici_bandwidth_gbyte_s",
        )
        print(
            f"ppermute_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
            f"{matrix_size_gbyte=}, achieved_bandwidth_gbyte_s (median) = {ici_bandwidth_gbyte_s_statistics.statistics['p50']}"
        )
        metrics.update(ici_bandwidth_gbyte_s_statistics.serialize_statistics())
    return metadata, metrics


def all_to_all_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str = None,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the all_to_all collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run.
      mesh_shape: The shape of the mesh.

    Returns:
      The measured time for the ICI benchmark.
    """
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
        "--xla_tpu_use_tc_device_shape_on_sc=true",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)
    key = jax.random.key(SEED)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)
    op_dimension_tuple = op_dimension.split("x")
    op_dimension_tuple = tuple(int(dim) for dim in op_dimension_tuple)
    sharding_axis = tuple(
        name
        for i, name in enumerate(mesh.axis_names)
        if op_dimension_tuple[i] > 1
    )

    def f(x):
        with jax.named_scope(MARKER):
            return jax.lax.all_to_all(x, sharding_axis, split_axis=0, concat_axis=0, tiled=True)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=lhs_sharding.spec,
            out_specs=out_sharding,
            check_rep=False,
        )
    )
    m = matrix_dim
    n = BASE_SHAPE[1]
    k = BASE_SHAPE[2]

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'

        matrix = jnp.ones((m, n, k), dtype=dtype)
        return (matrix,)

    print("Running all_to_all benchmark", num_runs, matrix_dim)
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="all_to_all_ici_op",
        trace_dir=trace_dir,
    )
    return {
        "ici_average_time_ms_list": time_ms_list,
    }


def all_to_all_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_to_all benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    input_num_elements = matrix_dim * BASE_SHAPE[1] * BASE_SHAPE[2]
    metadata.update({
        "input_num_elements": input_num_elements,
        "matrix_shape": json.dumps(f"({matrix_dim}, {BASE_SHAPE[1]}, {BASE_SHAPE[2]})"),
    })
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
