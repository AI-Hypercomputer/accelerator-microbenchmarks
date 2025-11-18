"""A script to run the microbenchmarks in Jax over DCN and ICI collectives."""

# pylint: disable=g-importing-member
from functools import partial
from typing import Any, Dict
import os

from benchmark_utils import simple_timeit, MetricsStatistics
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from common import MARKER
# pylint: disable=g-importing-member

BASE_SHAPE=[1,8,128]

def create_mesh(
    ici_size: int, mesh_shape: str
) -> tuple[Mesh, list[int], list[int]]:
    """Creates a mesh with the given ICI size."""
    ici_parallelism = [1, ici_size]

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
    mesh_devices = mesh_utils.create_device_mesh(shape, devices=jax.devices())
    mesh = Mesh(mesh_devices, axis_names)
    return mesh, ici_parallelism


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
    mesh = create_mesh(ici_size, mesh_shape)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    ici_average_time_ms_list = None
    # ICI benchmark
    if ici_size > 1:

        @partial(shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, None))
        def f(x):
            return jax.lax.psum(x, "ici")

        sharded_matrix = jax.device_put(
            matrix, jax.sharding.NamedSharding(mesh, P(None, None))
        )
        jitted_op = jax.jit(f)
        ici_average_time_ms_list = simple_timeit(
            jitted_op,
            sharded_matrix,
            matrix_dim=matrix_dim,
            tries=num_runs,
            task="psum_ici_op",
            trace_dir=trace_dir,
        )
    return {
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def psum_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_average_time_ms_list is not None:
        # bandwidth is claculated as psum can be done via reduce_scatter +
        # all_gather so bandwidth is the sum of the two (formulas below)
        ici_bandwidth_gbyte_s_list = [
            matrix_size_gbyte
            * (ici_size - 1)
            * 2
            / ici_size
            / (ici_average_time_ms / 1e3)
            for ici_average_time_ms in ici_average_time_ms_list
        ]
        ici_bandwidth_gbyte_s_statistics = MetricsStatistics(
            metrics_list=ici_bandwidth_gbyte_s_list,
            metrics_name="ici_bandwidth_gbyte_s",
        )
        print(
            f"psum_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
            f"{matrix_size_gbyte=}, achieved_bandwidth_gbyte_s (median) = {ici_bandwidth_gbyte_s_statistics.statistics['p50']}"
        )
        # Gather the metrics to report.
        metrics.update(ici_bandwidth_gbyte_s_statistics.serialize_statistics())
    return metadata, metrics


def psum_scatter_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
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
    mesh = create_mesh(ici_size, mesh_shape)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    ici_average_time_ms_list = None

    # ICI benchmark
    if ici_size > 1:

        @partial(shard_map, mesh=mesh, in_specs=P(None, None), out_specs=P(None, "ici"))
        def f(x):
            return jax.lax.psum_scatter(x, "ici", tiled=True)

        sharded_matrix = jax.device_put(
            matrix, jax.sharding.NamedSharding(mesh, P(None, None))
        )
        jitted_op = jax.jit(f)
        ici_average_time_ms_list = simple_timeit(
            jitted_op,
            sharded_matrix,
            matrix_dim=matrix_dim,
            tries=num_runs,
            task="psum_scatter_ici_op",
            trace_dir=trace_dir,
        )

    return {
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def psum_scatter_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum_scatter benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_average_time_ms_list is not None:
        # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
        # to use (ici_size - 1) steps in a ring algorithm
        ici_bandwidth_gbyte_s_list = [
            matrix_size_gbyte * (ici_size - 1) / ici_size / (ici_average_time_ms / 1e3)
            for ici_average_time_ms in ici_average_time_ms_list
        ]
        ici_bandwidth_gbyte_s_statistics = MetricsStatistics(
            metrics_list=ici_bandwidth_gbyte_s_list,
            metrics_name="ici_bandwidth_gbyte_s",
        )
        print(
            f"psum_scatter_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
            f"{matrix_size_gbyte=}, achieved_bandwidth_gbyte_s (median) = {ici_bandwidth_gbyte_s_statistics.statistics['p50']}"
        )
        # Gather the metrics to report.
        metrics.update(ici_bandwidth_gbyte_s_statistics.serialize_statistics())
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

    matrix = jnp.ones((matrix_dim, BASE_SHAPE[1], BASE_SHAPE[2]), dtype=dtype)
    ici_average_time_ms_list = None

    # ICI benchmark
    if ici_size > 1:
        op_dimension = op_dimension.split("x")
        op_dimension = tuple(int(dim) for dim in op_dimension)
        sharding_axis = tuple(
            name for i, name in enumerate(mesh.axis_names) if op_dimension[i] > 1
        )

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=P(None, None, None),
            out_specs=P(None, None, None),
            check_rep=False,
        )
        def f(x):
            with jax.named_scope(MARKER):
                return jax.lax.all_gather(x, sharding_axis, tiled=True)

        sharded_matrix = jax.device_put(
            matrix, jax.sharding.NamedSharding(mesh, P(None, None, None))
        )
        jitted_op = jax.jit(f)
        ici_average_time_ms_list = simple_timeit(
            jitted_op,
            sharded_matrix,
            matrix_dim=matrix_dim,
            tries=num_runs,
            task="all_gather_ici_op",
            trace_dir=trace_dir,
        )

    return {
        "ici_average_time_ms_list": ici_average_time_ms_list,
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
    dtype_bytes = dtype.dtype.itemsize
    metadata.update({
        "input_num_elements": input_num_elements,
        "dtype_bytes": dtype_bytes,
        "matrix_shape": f"({matrix_dim}, 8, 128)",
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
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
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
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9

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
    mesh = create_mesh(ici_size, mesh_shape)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    ici_average_time_ms_list = None

    # ICI benchmark
    if ici_size > 1:

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=P(None, None),
            out_specs=P(None, None),
            check_rep=False,
        )
        def f(x):
            return jax.lax.all_to_all(x, "ici", split_axis=0, concat_axis=0)

        sharded_matrix = jax.device_put(
            matrix, jax.sharding.NamedSharding(mesh, P(None, None))
        )
        jitted_op = jax.jit(f)
        ici_average_time_ms_list = simple_timeit(
            jitted_op,
            sharded_matrix,
            matrix_dim=matrix_dim,
            tries=num_runs,
            task="all_to_all_ici_op",
            trace_dir=trace_dir,
        )

    return {
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def all_to_all_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    ici_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_to_all benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_average_time_ms_list is not None:
        ici_bandwidth_gbyte_s_list = [
            matrix_size_gbyte * (ici_size - 1) / ici_size / (ici_average_time_ms / 1e3)
            for ici_average_time_ms in ici_average_time_ms_list
        ]
        ici_bandwidth_gbyte_s_statistics = MetricsStatistics(
            metrics_list=ici_bandwidth_gbyte_s_list,
            metrics_name="ici_bandwidth_gbyte_s",
        )
        print(
            f"all_to_all_ici: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
            f"{matrix_size_gbyte=}, achieved_bandwidth_gbyte_s (median) = {ici_bandwidth_gbyte_s_statistics.statistics['p50']}"
        )
        # Gather the metrics to report.
        metrics.update(ici_bandwidth_gbyte_s_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
