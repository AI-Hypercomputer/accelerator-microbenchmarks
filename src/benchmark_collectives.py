"""A script to run the microbenchmarks in Jax over DCN and ICI collectives."""

# pylint: disable=g-importing-member
from functools import partial
from typing import Any, Dict, Tuple

from benchmark_utils import simple_timeit, MetricsStatistics
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# pylint: disable=g-importing-member


def create_mesh(dcn_size: int, ici_size: int) -> tuple[Mesh, list[int], list[int]]:
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
        mesh_devices = mesh_utils.create_device_mesh([ici_size], devices=jax.devices())
        mesh = Mesh(mesh_devices, "ici")
    return mesh


def extract_metadata(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_keys = ["ici_time_ms_list", "dcn_time_ms_list"]
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_keys
    }
    metadata["dtype"] = metadata["dtype"].dtype.itemsize
    return metadata

def generate_metrics_statistics(
    metrics_list: list[float],
    metrics_name: str,
    benchmark_name: str,
    matrix_dim: int,
    dtype: Any,
    matrix_size_gbyte: float,
    metrics: Dict[str, Any],
) -> None:
    """Calculates statistics for a metrics list, prints p50, and updates the metrics dict."""
    if not metrics_list:
        return
    statistics = MetricsStatistics(
        metrics_list=metrics_list,
        metrics_name=metrics_name,
    )
    print(
        f"{benchmark_name}: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, {metrics_name} (median) = {statistics.statistics['p50']}"
    )
    metrics.update(statistics.serialize_statistics())

def benchmark_collective(
    benchmark_name: str,
    jax_op: Any,
    mesh: Mesh,
    matrix: jnp.ndarray,
    matrix_dim: int,
    axis_name: str,
    in_specs: P,
    out_specs: P,
    check_rep: bool = True,
    jax_op_kwargs: Dict[str, Any] = None,
    num_runs: int = 1,
    warmup_tries: int = 10,
    trace_dir: str = None,
) -> list[float]:
    """
    Helper function to run a collective benchmark on DCN and ICI.

    Args:
      benchmark_name: The base name for the benchmark task.
      jax_op: The JAX collective operation to benchmark (e.g., jax.lax.psum).
      mesh: The JAX device mesh to run the benchmark on.
      matrix: The input array for the collective operation.
      matrix_dim: The dimension of the input matrix.
      axis_name: The name of the axis over which the op is performed (e.g., "dcn" or "ici").
      in_specs: The input sharding specs.
      out_specs: The output sharding specs.
      check_rep: Indicate if replication check is needed. Can be skipped in some situations.
      jax_op_kwargs: Optional keyword arguments for the JAX operation.
      num_runs: The number of times to run the benchmark operation for timing.
      warmup_tries: The number of warmup runs before the actual timing.
      trace_dir: Optional directory to save JAX traces.

    Returns:
      A list of time in milliseconds for each run.
    """
    if jax_op_kwargs is None:
        jax_op_kwargs = {}
    if axis_name != "dcn" and axis_name != "ici":
        raise ValueError(f"Unsupported axis name: {axis_name}")

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=check_rep,
    )
    def f(x):
        return jax_op(x, axis_name, **jax_op_kwargs)

    jitted_op = jax.jit(f)
    time_ms_list = simple_timeit(
        jitted_op,
        matrix,
        matrix_dim=matrix_dim,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task=f"{benchmark_name}_{axis_name}_op",
        trace_dir=trace_dir,
    )

    return time_ms_list

def psum_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
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
    mesh = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    results = {
        "dcn_time_ms_list": None,
        "ici_time_ms_list": None,
    }

    if dcn_size > 1:
        results["dcn_time_ms_list"] = benchmark_collective(
            benchmark_name="psum", jax_op=jax.lax.psum, mesh=mesh, matrix=matrix,
            matrix_dim=matrix_dim, axis_name="dcn", in_specs=P("dcn", None),
            out_specs=P(None, None), num_runs=num_runs, warmup_tries=warmup_tries,
            trace_dir=trace_dir,
        )

    if ici_size > 1:
        results["ici_time_ms_list"] = benchmark_collective(
            benchmark_name="psum", jax_op=jax.lax.psum, mesh=mesh, matrix=matrix,
            matrix_dim=matrix_dim, axis_name="ici", in_specs=P(None, None),
            out_specs=P(None, None), num_runs=num_runs, warmup_tries=warmup_tries,
            trace_dir=trace_dir,
        )

    return results

def psum_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_time_ms_list: list[float],
    dcn_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = extract_metadata(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1 and dcn_time_ms_list is not None:
        # bandwidth is claculated as psum can be done via reduce_scatter +
        # all_gather so bandwidth is the sum of the two (formulas below)
        dcn_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (dcn_size - 1)
                * 2
                / dcn_size
                / dcn_size
                / (dcn_time_ms / 1e3)
                for dcn_time_ms in dcn_time_ms_list
        ]
        generate_metrics_statistics(
            dcn_bandwidth_gbyte_s_list, "dcn_bandwidth_gbyte_s", "psum_dcn",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_time_ms_list is not None:
        # bandwidth is claculated as psum can be done via reduce_scatter +
        # all_gather so bandwidth is the sum of the two (formulas below)
        ici_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (ici_size - 1)
                * 2
                / ici_size
                / (ici_time_ms / 1e3)
                for ici_time_ms in ici_time_ms_list
        ]
        generate_metrics_statistics(
            ici_bandwidth_gbyte_s_list, "ici_bandwidth_gbyte_s", "psum_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        generate_metrics_statistics(
            ici_time_ms_list, "ici_time_ms", "psum_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics)
        metrics["ici_time_ms_list"] = ici_time_ms_list

    return metadata, metrics

def psum_scatter_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
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
    mesh = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    results = {
        "dcn_time_ms_list": None,
        "ici_time_ms_list": None,
    }

    psum_scatter_kwargs = {"tiled": True}


    if dcn_size > 1:
        results["dcn_time_ms_list"] = benchmark_collective(
            benchmark_name="psum_scatter", jax_op=jax.lax.psum_scatter, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="dcn",
            in_specs=P("dcn", None), out_specs=P("dcn", None),
            jax_op_kwargs=psum_scatter_kwargs, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    if ici_size > 1:
        results["ici_time_ms_list"] = benchmark_collective(
            benchmark_name="psum_scatter", jax_op=jax.lax.psum_scatter, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="ici",
            in_specs=P(None, None), out_specs=P(None, "ici"),
            jax_op_kwargs=psum_scatter_kwargs, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    return results

def psum_scatter_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_time_ms_list: list[float],
    dcn_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum_scatter benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = extract_metadata(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1 and dcn_time_ms_list is not None:

        # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
        # to use (dcn_size - 1) steps in a ring algorithm
        dcn_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (dcn_size - 1)
                / dcn_size
                / dcn_size
                / (dcn_time_ms / 1e3)
                for dcn_time_ms in dcn_time_ms_list
        ]
        generate_metrics_statistics(
            dcn_bandwidth_gbyte_s_list, "dcn_bandwidth_gbyte_s", "psum_scatter_dcn",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_time_ms_list is not None:
        # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
        # to use (ici_size - 1) steps in a ring algorithm
        ici_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (ici_size - 1)
                / ici_size
                / (ici_time_ms / 1e3)
                for ici_time_ms in ici_time_ms_list
        ]
        generate_metrics_statistics(
            ici_bandwidth_gbyte_s_list, "ici_bandwidth_gbyte_s", "psum_scatter_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        generate_metrics_statistics(
            ici_time_ms_list, "ici_time_ms", "psum_scatter_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        metrics["ici_time_ms_list"] = ici_time_ms_list

    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def all_gather_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    warmup_tries: int = 10,
    num_runs: int = 1,
    trace_dir: str = None,
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
    mesh = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    results = {
        "dcn_time_ms_list": None,
        "ici_time_ms_list": None,
    }

    all_gather_kwargs = {"tiled": True}

    if dcn_size > 1:
        results["dcn_time_ms_list"] = benchmark_collective(
            benchmark_name="all_gather", jax_op=jax.lax.all_gather, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="dcn",
            in_specs=P("dcn", None), out_specs=P(None, None), check_rep=False,
            jax_op_kwargs=all_gather_kwargs, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    if ici_size > 1:
        results["ici_time_ms_list"] = benchmark_collective(
            benchmark_name="all_gather", jax_op=jax.lax.all_gather, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="ici",
            in_specs=P("ici", None), out_specs=P(None, None), check_rep=False,
            jax_op_kwargs=all_gather_kwargs, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    return results

def all_gather_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_time_ms_list: list[float],
    dcn_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_gather benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = extract_metadata(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1 and dcn_time_ms_list is not None:

        # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
        # to use (dcn_size - 1) steps in a ring algorithm
        dcn_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (dcn_size - 1)
                / dcn_size
                / (dcn_time_ms / 1e3)
                for dcn_time_ms in dcn_time_ms_list
        ]
        generate_metrics_statistics(
            dcn_bandwidth_gbyte_s_list, "dcn_bandwidth_gbyte_s", "all_gather_dcn",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_time_ms_list is not None:
        # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
        # to use (ici_size - 1) steps in a ring algorithm
        ici_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (ici_size - 1)
                / ici_size
                / (ici_time_ms / 1e3)
                for ici_time_ms in ici_time_ms_list
        ]
        generate_metrics_statistics(
            ici_bandwidth_gbyte_s_list, "ici_bandwidth_gbyte_s", "all_gather_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        generate_metrics_statistics(
            ici_time_ms_list, "ici_time_ms", "all_gather_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        metrics["ici_time_ms_list"] = ici_time_ms_list

    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics

def ppermute_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
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
    mesh = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    results = {
        "dcn_time_ms_list": None,
        "ici_time_ms_list": None,
    }


    if dcn_size > 1:
        dcn_perm = [(i, (i + 1) % dcn_size) for i in range(dcn_size)]
        results["dcn_time_ms_list"] = benchmark_collective(
            benchmark_name="ppermute", jax_op=jax.lax.ppermute, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="dcn",
            in_specs=P("dcn", None), out_specs=P("dcn", None),
            jax_op_kwargs={"perm": dcn_perm}, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    if ici_size > 1:
        ici_perm = [(i, (i + 1) % ici_size) for i in range(ici_size)]
        results["ici_time_ms_list"] = benchmark_collective(
            benchmark_name="ppermute", jax_op=jax.lax.ppermute, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="ici",
            in_specs=P(None, None), out_specs=P(None, "ici"),
            jax_op_kwargs={"perm": ici_perm}, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    return results

def ppermute_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_time_ms_list: list[float],
    dcn_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the ppermute benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = extract_metadata(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1 and dcn_time_ms_list is not None:

        # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
        # to use 1 step
        dcn_bandwidth_gbyte_s_list = [
                matrix_size_gbyte / dcn_size / (dcn_time_ms / 1e3)
                for dcn_time_ms in dcn_time_ms_list
        ]
        generate_metrics_statistics(
            dcn_bandwidth_gbyte_s_list, "dcn_bandwidth_gbyte_s", "ppermute_dcn",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_time_ms_list is not None:
        # each sharded matrix size is matrix_size_gbyte / ici_size and then it needs
        # to use 1 step
        ici_bandwidth_gbyte_s_list = [
            matrix_size_gbyte / (ici_time_ms / 1e3)
            for ici_time_ms in ici_time_ms_list
        ]
        generate_metrics_statistics(
            ici_bandwidth_gbyte_s_list, "ici_bandwidth_gbyte_s", "ppermute_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        generate_metrics_statistics(
            ici_time_ms_list, "ici_time_ms", "ppermute_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        metrics["ici_time_ms_list"] = ici_time_ms_list
    return metadata, metrics

def all_to_all_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
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
    mesh = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    results = {
        "dcn_time_ms_list": None,
        "ici_time_ms_list": None,
    }

    all_to_all_kwargs = {"split_axis": 0, "concat_axis": 0, "tiled": True}

    if dcn_size > 1:
        results["dcn_time_ms_list"] = benchmark_collective(
            benchmark_name="all_to_all", jax_op=jax.lax.all_to_all, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="dcn",
            in_specs=P("dcn", None), out_specs=P("dcn", None),
            jax_op_kwargs=all_to_all_kwargs, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    if ici_size > 1:
        results["ici_time_ms_list"] = benchmark_collective(
            benchmark_name="all_to_all", jax_op=jax.lax.all_to_all, mesh=mesh,
            matrix=matrix, matrix_dim=matrix_dim, axis_name="ici",
            in_specs=P(None, None), out_specs=P(None, None), check_rep=False,
            jax_op_kwargs=all_to_all_kwargs, num_runs=num_runs,
            warmup_tries=warmup_tries, trace_dir=trace_dir,
        )

    return results

def all_to_all_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_time_ms_list: list[float],
    dcn_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_to_all benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = extract_metadata(params)
    metrics = {}
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1 and dcn_time_ms_list is not None:

        dcn_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (dcn_size - 1)
                / dcn_size
                / dcn_size
                / (dcn_time_ms / 1e3)
                for dcn_time_ms in dcn_time_ms_list
        ]
        generate_metrics_statistics(
            dcn_bandwidth_gbyte_s_list, "dcn_bandwidth_gbyte_s", "all_to_all_dcn",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1 and ici_time_ms_list is not None:
        ici_bandwidth_gbyte_s_list = [
                matrix_size_gbyte
                * (ici_size - 1)
                / ici_size
                / (ici_time_ms / 1e3)
                for ici_time_ms in ici_time_ms_list
        ]
        generate_metrics_statistics(
            ici_bandwidth_gbyte_s_list, "ici_bandwidth_gbyte_s", "all_to_all_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        generate_metrics_statistics(
            ici_time_ms_list, "ici_time_ms", "all_to_all_ici",
            matrix_dim, dtype, matrix_size_gbyte, metrics
        )
        metrics["ici_time_ms_list"] = ici_time_ms_list

    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
