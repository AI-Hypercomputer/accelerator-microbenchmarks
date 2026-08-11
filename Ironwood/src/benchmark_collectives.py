"""A script to run the microbenchmarks in Jax over ICI collectives."""

import json
import math
import os
from typing import Any, Dict

from benchmark_utils import find_sparsecore_usage_from_xplane
from benchmark_utils import get_lhs_named_shading
from benchmark_utils import get_out_sharding
from benchmark_utils import MetricsStatistics
from benchmark_utils import multiple_iteration_timeit_from_trace
from benchmark_utils import ShardingStrategy
from benchmark_utils import get_real_dtype_bytes
from common import MARKER
import jax
from jax import core
from jax import ffi
from jax._src.core import Primitive
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.interpreters import mlir
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

BASE_SHAPE = [1, 8, 128]
SEED = 0
GLOBAL_SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
GLOBAL_PSTATE = 7
LOG_SPARSECORE_USAGE = False
V6E_DEVICE_KINDS = ["TPU v6 lite", "TPU v6e"]
PEAK_ICI_LINK_BW = 100.0  # GB/s unidirectional for both v6e and tpu7x


def calculate_all_to_all_theoretical_bandwidth(
    topology: str,
    device_kind: str,
    rank: int,
    participating_ranks: int,
) -> Any:
    """Calculates theoretical peak All-to-All bandwidth in GB/s."""
    if (
        not topology
        or rank is None
        or rank <= 0
        or participating_ranks is None
        or participating_ranks <= 0
    ):
        return None

    try:
        topo_dims = [
            int(x) for x in topology.split("x") if x.strip() and int(x) > 0
        ]
    except (ValueError, AttributeError):
        return None

    if not topo_dims:
        return None

    spatial_dims = [d for d in topo_dims if d > 1] or [1]
    dim_count = len(spatial_dims)

    if device_kind in V6E_DEVICE_KINDS:
        # TPU v6e (2D): wrap-around link exists if any dimension >= 16
        has_wrap = any(d >= 16 for d in topo_dims)
    else:
        # TPU v7x (3D): wrap-around torus requires all 3 spatial dimensions >= 4
        has_wrap = (len(topo_dims) >= 3) and all(d >= 4 for d in topo_dims[:3])

    wrap_factor = 2.0 if has_wrap else 1.0
    sorted_dims = sorted(spatial_dims)

    if dim_count == 1:
        link_multiplier = 1.0 * wrap_factor
    elif dim_count == 2:
        link_multiplier = sorted_dims[0] * wrap_factor
    else:
        link_multiplier = sorted_dims[0] * sorted_dims[1] * wrap_factor

    # Generic All-To-All Formula :
    # 4 * link_bw (100.0) * link_multiplier * participating_ranks / (rank^2)
    theo_bw = (
        4.0
        * PEAK_ICI_LINK_BW
        * link_multiplier
        * float(participating_ranks)
        / (float(rank) * float(rank))
    )
    return theo_bw


def create_mesh(ici_size: int, mesh_shape: str) -> Mesh:
    """Creates a mesh with the given ICI size."""
    devices_needed = ici_size
    local_devices = jax.local_devices()
    if devices_needed <= len(local_devices):
        devices = local_devices
    else:
        devices = jax.devices()

    if len(devices) < devices_needed:
        raise ValueError(
            f"Need {devices_needed} devices, but found {len(devices)}"
        )
    devices = devices[:devices_needed]
    mesh_shape = mesh_shape.split("x")
    mesh_shape = [int(i) for i in mesh_shape]

    shape = mesh_shape if mesh_shape else (ici_size,)

    axis_names = [f"d_{i}" for i in range(len(shape))]

    first_device = devices[0]
    device_kind = first_device.device_kind
    print("Device kind: ", device_kind)
    mesh_devices = mesh_utils.create_device_mesh(shape, devices=devices)
    mesh = Mesh(mesh_devices, axis_names)
    return mesh


def get_sharding_axis(dim_str: str, mesh: Mesh) -> tuple[str, ...]:
    """Computes sharding axis names from dimension string and mesh."""
    # Example of a dimension string is '1x4'
    dim_tuple = dim_str.split("x")
    dim_tuple = tuple(int(dim) for dim in dim_tuple)
    sharding_axis = tuple(
        name for i, name in enumerate(mesh.axis_names) if dim_tuple[i] > 1
    )
    return sharding_axis


def get_metrics_helper(
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_keys = ["ici_average_time_ms", "xla_output"]
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_keys
    }
    metadata["dtype"] = get_real_dtype_bytes(metadata["dtype"].dtype)
    return metadata


def unified_ici_collectives_metrics(
    xla_output: str,
    matrix_shape: tuple[int, int, int],
    dtype: jnp.dtype,
    mesh_shape: str,
    op_dimension: str,
    sharding_strategy: str,
    ici_average_time_ms_list: list[float],
    iteration: int,
    op_type: str,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    """Calculates the metrics for the ICI collectives benchmark."""

    average_time_ms_statistics = MetricsStatistics(
        metrics_list=ici_average_time_ms_list, metrics_name="step_time_ms"
    )
    hlo_input_shape = hlo_output_shape = hlo_replica_groups = None
    hlo_first_replica_group = []

    input_num_elements = matrix_shape[0] * matrix_shape[1] * matrix_shape[2]
    dtype_bytes = get_real_dtype_bytes(dtype.dtype)
    if xla_output:
        xla_output_json = json.loads(xla_output)
        hlo_input_shape = xla_output_json.get("hlo_input_shape")
        hlo_output_shape = xla_output_json.get("hlo_output_shape")
        hlo_replica_groups = xla_output_json.get("hlo_replica_groups")
        hlo_first_replica_group = xla_output_json.get("hlo_first_replica_group")

    rank = max(len(hlo_first_replica_group), 1)

    if all(i % 2 == 0 for i in hlo_first_replica_group):
        replica_group_type = "parallel"
    else:
        replica_group_type = "non-parallel"

    # Safe to access [0] without safeguard because JAX guarantees at least
    # one device is always initialized (CPU fallback if no accelerator), and
    # mesh creation has already validated that the requested number of
    # devices exist.
    device_kind = jax.devices()[0].device_kind
    if device_kind in V6E_DEVICE_KINDS:
        # For TPU v6e (Trillium), 1 physical chip = 1 JAX device.
        # Ring collective communication volume per chip across N ranks is
        # exactly (N - 1) shards.
        # There is no dual-core traffic multiplier needed.
        participating_ranks = rank - 1
        tf_multiplier = 1
    else:
        # Dual-core logic for TPU v7x
        if replica_group_type == "parallel":
            participating_ranks = rank - 1
            tf_multiplier = 2
        else:
            participating_ranks = rank - 2
            tf_multiplier = 1

    transferred_data = 0
    if op_type == "AG":
        transferred_data = (
            input_num_elements
            * participating_ranks
            * dtype_bytes
            * 0.000000001
            * tf_multiplier
        )
    elif op_type == "AR":
        transferred_data = (
            input_num_elements
            * participating_ranks
            * dtype_bytes
            * 0.000000001
            * tf_multiplier
            * 2
            / rank
        )
    elif op_type in ["RS", "A2A"]:
        transferred_data = (
            input_num_elements
            * participating_ranks
            * dtype_bytes
            * 0.000000001
            * tf_multiplier
            / rank
        )

    # Calculate All-to-All Theoretical Peak Bandwidth if topology is provided
    theo_bw = None
    if op_type == "A2A" and topology:
        theo_bw = calculate_all_to_all_theoretical_bandwidth(
            topology=topology,
            device_kind=device_kind,
            rank=rank,
            participating_ranks=participating_ranks,
        )

    has_valid_theo_bw = theo_bw is not None and theo_bw > 0

    sparsecore_used = "NA"
    if LOG_SPARSECORE_USAGE:
        print("trace_dir: ", trace_dir)
        if trace_dir:
            sparsecore_used = find_sparsecore_usage_from_xplane(trace_dir)
        print("sparsecore_used: ", sparsecore_used)
    print("hlo first replica group: ", hlo_first_replica_group)

    metadata = {
        "iteration": iteration,
        "op_type": op_type,
        "replica_group_type": replica_group_type,
        "rank": rank,
        "mesh_shape": mesh_shape,
        "op_dimension": op_dimension,
        "sharding_strategy": sharding_strategy,
        "input_num_elements": input_num_elements,
        "matrix_shape": json.dumps(f"({matrix_shape})"),
        "transferred_data (GB)": transferred_data,
        "dtype_bytes": dtype_bytes,
        "hlo_input_shape": json.dumps(hlo_input_shape),
        "hlo_output_shape": json.dumps(hlo_output_shape),
        "hlo_replica_groups": json.dumps(hlo_replica_groups),
        "sparsecore_used": sparsecore_used,
    }
    if topology:
        metadata["topology"] = topology
    if has_valid_theo_bw:
        metadata["theoretical_bw (GB/s)"] = theo_bw
    achieved_bw = [
        transferred_data * 1000 / my_time
        for my_time in ici_average_time_ms_list
    ]
    achieved_bw_statistics = MetricsStatistics(
        metrics_list=achieved_bw, metrics_name="achieved_bw (GB/s)"
    )
    metrics = {}
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(achieved_bw_statistics.serialize_statistics())

    if has_valid_theo_bw:
        utilization_list = [(bw / theo_bw) * 100.0 for bw in achieved_bw]
        util_stats = MetricsStatistics(
            metrics_list=utilization_list, metrics_name="utilization_pct"
        )
        metrics.update(util_stats.serialize_statistics())

    print("metadata: ", metadata)
    print("metrics: ", metrics)
    return metadata, metrics


def psum_benchmark(
    matrix_dim: int,
    mesh_shape: str,
    sharding_strategy: str,
    op_dimension: int = 1,
    ici_size: int = 1,
    dtype: jnp.dtype = jax.numpy.bfloat16,
    num_runs: int = 1,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Benchmarks the psum collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      mesh_shape: The shape of the mesh.
      op_dimension: The dimension of the operation.
      ici_size: The number of chips in a single slice. If 1, then no ICI
        benchmark is run.
      dtype: The data type of the matrix.
      num_runs: The number of runs to perform.
      trace_dir: The directory to save the trace to.

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
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)
    key = jax.random.key(SEED)
    lhs_sharding = get_lhs_named_shading(mesh, GLOBAL_SHARDING_STRATEGY)
    out_sharding = get_out_sharding(GLOBAL_SHARDING_STRATEGY)
    sharding_axis = get_sharding_axis(sharding_strategy, mesh)

    # 1. Define the Primitive
    zero_crop_p = Primitive("zero_crop")

    # 2. Implement Abstract Evaluation (output shape/dtype is same as input)
    def zero_crop_abstract_eval(x):
        return core.ShapedArray(x.shape, x.dtype)

    zero_crop_p.def_abstract_eval(zero_crop_abstract_eval)

    # 3. Implement the Lowering Rule using jax.ffi
    def zero_crop_lowering(ctx, x):
        (aval_in,) = ctx.avals_in
        (aval_out,) = ctx.avals_out

        return ffi.ffi_lowering(
            "ZeroCrop",
            operands=[x],
            operand_layouts=mlir.default_layouts(ctx, aval_in),
            result_layouts=mlir.default_layouts(ctx, aval_out),
        )(ctx, x)

    mlir.register_lowering(zero_crop_p, zero_crop_lowering)

    # 4. Create a Python Wrapper using jax.ffi.ffi_call
    def zero_crop(x):
        return ffi.ffi_call(
            "ZeroCrop",
            result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
            has_side_effect=True,
        )(x)

    def f(x):
        with jax.named_scope(MARKER):
            y = jax.lax.psum(x, sharding_axis)
            # Insert the custom call to prevent y from being a live out buffer
            return zero_crop(y)

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
        task="psum_ici_op",
        trace_dir=trace_dir,
    )
    return {
        "ici_average_time_ms_list": time_ms_list,
        "matrix_shape": (m, n, k),
        "op_type": "AR",
        "trace_dir": trace_dir,
    }


def psum_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    sharding_strategy: str,
    ici_average_time_ms_list: list[float],
    matrix_shape: tuple[int, int, int],
    xla_output: str,
    op_type: str,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Calculates the metrics for the psum benchmark."""
    # Build dictionary of all the parameters in the function

    return unified_ici_collectives_metrics(
        xla_output,
        matrix_shape,
        dtype,
        mesh_shape,
        op_dimension,
        sharding_strategy,
        ici_average_time_ms_list,
        matrix_dim,
        op_type,
        trace_dir=trace_dir,
        topology=topology,
    )


def psum_scatter_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    sharding_strategy: str = None,
    op_dimension: int = 1,
    num_runs: int = 1,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Benchmarks the psum_scatter collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI
        benchmark is run.
      mesh_shape: The shape of the mesh.
      op_dimension: The dimension of the operation.
      sharding_strategy: The sharding strategy of the operation.
      num_runs: The number of runs to perform.
      trace_dir: The directory to save the trace to.

    Returns:
      The measured time for the ICI benchmark.
    """
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
        "--xla_tpu_enable_reduce_scatter_offload_tracing=true",
        "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",  # pylint: disable=line-too-long
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true",
        "--xla_tpu_use_tc_device_shape_on_sc=true",
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)

    sharding_axis = get_sharding_axis(sharding_strategy, mesh)

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
    sharding_strategy_tuple = tuple(map(int, sharding_strategy.split("x")))
    op_dimension_tuple_multiplier = math.prod(sharding_strategy_tuple)
    m = op_dimension_tuple_multiplier
    n = matrix_dim
    k = 256

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
    print("Matrix shape: ", m, n, k)
    return {
        "ici_average_time_ms_list": time_ms_list,
        "matrix_shape": (m, n, k),
        "op_type": "RS",
        "trace_dir": trace_dir,
    }


def psum_scatter_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    sharding_strategy: str,
    ici_average_time_ms_list: list[float],
    matrix_shape: tuple[int, int, int],
    xla_output: str,
    op_type: str,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Calculates the metrics for the psum_scatter benchmark."""
    # Build dictionary of all the parameters in the function

    return unified_ici_collectives_metrics(
        xla_output,
        matrix_shape,
        dtype,
        mesh_shape,
        op_dimension,
        sharding_strategy,
        ici_average_time_ms_list,
        matrix_dim,
        op_type,
        trace_dir=trace_dir,
        topology=topology,
    )


def all_gather_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    sharding_strategy: str,
    op_dimension: int = 1,
    num_runs: int = 1,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Benchmarks the all_gather collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI
        benchmark is run.
      mesh_shape: The shape of the mesh.
      sharding_strategy: The sharding strategy of the operation.
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
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
        "--xla_tpu_scoped_vmem_limit_kib=65536",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)

    sharding_axis = get_sharding_axis(sharding_strategy, mesh)

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
        "matrix_shape": (m, n, k),
        "op_type": "AG",
        "trace_dir": trace_dir,
    }


def all_gather_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    sharding_strategy: str,
    ici_average_time_ms_list: list[float],
    matrix_shape: tuple[int, int, int],
    xla_output: str,
    op_type: str,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Calculates the metrics for the all_gather benchmark."""
    # Build dictionary of all the parameters in the function

    return unified_ici_collectives_metrics(
        xla_output,
        matrix_shape,
        dtype,
        mesh_shape,
        op_dimension,
        sharding_strategy,
        ici_average_time_ms_list,
        matrix_dim,
        op_type,
        trace_dir=trace_dir,
        topology=topology,
    )


def all_to_all_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    sharding_strategy: str,
    op_dimension: int = 1,
    num_runs: int = 1,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Benchmarks the all_to_all collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      ici_size: The number of chips in a single slice. If 1, then no ICI
        benchmark is run.
      mesh_shape: The shape of the mesh.
      op_dimension: The dimension of the operation.
      num_runs: The number of runs to perform.
      trace_dir: The directory to save the trace to.

    Returns:
      The measured time for the ICI benchmark.
    """
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    mesh = create_mesh(ici_size, mesh_shape)
    key = jax.random.key(SEED)
    lhs_sharding = get_lhs_named_shading(mesh, GLOBAL_SHARDING_STRATEGY)
    out_sharding = get_out_sharding(GLOBAL_SHARDING_STRATEGY)
    sharding_axis = get_sharding_axis(sharding_strategy, mesh)

    def f(x):
        with jax.named_scope(MARKER):
            return jax.lax.all_to_all(
                x, sharding_axis, split_axis=0, concat_axis=0, tiled=True
            )

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
        "matrix_shape": (m, n, k),
        "op_type": "A2A",
        "trace_dir": trace_dir,
    }


def all_to_all_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str,
    op_dimension: str,
    sharding_strategy: str,
    ici_average_time_ms_list: list[float],
    matrix_shape: tuple[int, int, int],
    xla_output: str,
    op_type: str,
    trace_dir: str = None,
    topology: str = None,
) -> Dict[str, Any]:
    # pylint: disable=unused-argument
    """Calculates the metrics for the all_to_all benchmark."""
    # Build dictionary of all the parameters in the function

    return unified_ici_collectives_metrics(
        xla_output,
        matrix_shape,
        dtype,
        mesh_shape,
        op_dimension,
        sharding_strategy,
        ici_average_time_ms_list,
        matrix_dim,
        op_type,
        trace_dir=trace_dir,
        topology=topology,
    )
