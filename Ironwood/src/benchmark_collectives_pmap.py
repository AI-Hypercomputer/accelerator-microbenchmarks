"""A script to run the microbenchmarks in Jax over collectives using pmap.

This file provides pmap-based implementations of collectives to avoid
XLA compilation failures on multi-host allocations when JAX is initialized
in single-host mode. pmap collectives default to logical device IDs,
bypassing the compile-time physical ID checks that fail on non-zero hosts.
"""

import json
import math
import os
from typing import Any, Dict
import functools

import jax
from jax import core
from jax import ffi
from jax._src.core import Primitive
from jax.interpreters import mlir
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def device_put_sharded(shards, devices):
    """Drop-in replacement for jax.device_put_sharded supporting pytrees."""
    mesh = Mesh(np.array(devices), ('x',))
    sharding = NamedSharding(mesh, P('x'))
    stacked = np.stack([np.asarray(x) for x in shards])
    return jax.device_put(stacked, sharding)

def device_put_replicated(x, devices):
    """Drop-in replacement for jax.device_put_replicated supporting pytrees."""
    mesh = Mesh(np.array(devices), ('x',))
    sharding = NamedSharding(mesh, P()) # Empty P() means replicated
    return jax.device_put(x, sharding)

from benchmark_utils import multiple_iteration_timeit_from_trace
from benchmark_utils import get_real_dtype_bytes
from benchmark_collectives import unified_ici_collectives_metrics
from common import MARKER

BASE_SHAPE = [1, 8, 128]
SEED = 0
GLOBAL_PSTATE = 7


def psum_benchmark_pmap(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str = None,  # Kept for compatibility
    sharding_strategy: str = None,  # Kept for compatibility
    op_dimension: int = 1,  # Kept for compatibility
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the psum collective operation using pmap."""
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
        "--xla_tpu_pad_operations_input_tiles=true",
        "--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes=0",
        "--xla_tpu_use_tc_device_shape_on_sc=true",
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    
    devices = jax.local_devices()[:ici_size]
    
    # Define zero_crop to prevent compiler optimization
    zero_crop_p = Primitive("zero_crop")
    def zero_crop_abstract_eval(x):
        return core.ShapedArray(x.shape, x.dtype)
    zero_crop_p.def_abstract_eval(zero_crop_abstract_eval)
    
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
    
    def zero_crop(x):
        return ffi.ffi_call(
            "ZeroCrop",
            result_shape_dtypes=jax.ShapeDtypeStruct(x.shape, x.dtype),
            has_side_effect=True,
        )(x)

    # Named jit_f to match the hardcoded '*jit_f*' pattern in rename_xla_dump
    @functools.partial(jax.pmap, axis_name='i', devices=devices)
    def jit_f(x):
        with jax.named_scope(MARKER):
            y = jax.lax.psum(x, axis_name='i')
            return zero_crop(y)

    m = matrix_dim
    n = BASE_SHAPE[1]
    k = BASE_SHAPE[2]
    local_m = m // ici_size

    def data_generator():
        # Pre-distribute data to devices to avoid transfer overhead during timing
        arrays = [jnp.ones((local_m, n, k), dtype=dtype) for _ in range(ici_size)]
        sharded_input = device_put_sharded(arrays, devices)
        return (sharded_input,)

    print("Running psum pmap benchmark", num_runs, matrix_dim)
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_f,
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


def psum_benchmark_pmap_calculate_metrics(
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
    trace_dir: str,
) -> Dict[str, Any]:
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
        trace_dir,
    )


def psum_scatter_benchmark_pmap(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str = None,
    sharding_strategy: str = None,
    op_dimension: int = 1,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the psum_scatter collective operation using pmap."""
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        "--xla_sc_disable_megacore_partitioning=true",
        "--xla_tpu_disable_sparse_core_collective_offload_remover=true",
        "--xla_tpu_enable_reduce_scatter_offload_tracing=true",
        "--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true",
        "--xla_tpu_enable_sparse_core_reduce_scatter_v2=true",
        "--xla_tpu_use_tc_device_shape_on_sc=true",
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    
    devices = jax.local_devices()[:ici_size]
    
    # We assume sharding_strategy split size from the parameter
    # e.g., if strategy is "4x1x1", we split by 4.
    # For single-host, we can default to split by ici_size.
    split_factor = ici_size
    if sharding_strategy:
        try:
            split_factor = math.prod(map(int, sharding_strategy.split("x")))
        except ValueError:
            pass

    @functools.partial(jax.pmap, axis_name='i', devices=devices, in_axes=(None, 0))
    def jit_f(x, dummy):
        with jax.named_scope(MARKER):
            return jax.lax.psum_scatter(x, axis_name='i', scatter_dimension=0, tiled=True)

    m = split_factor
    n = matrix_dim
    k = 256

    def data_generator():
        matrix = jnp.ones((m, n, k), dtype=dtype)
        replicated_input = device_put_replicated(matrix, devices)
        dummy = device_put_sharded([jnp.array(i) for i in range(ici_size)], devices)
        return (replicated_input, dummy)

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="psum_scatter_ici_op",
        trace_dir=trace_dir,
    )
    
    return {
        "ici_average_time_ms_list": time_ms_list,
        "matrix_shape": (m, n, k),
        "op_type": "RS",
        "trace_dir": trace_dir,
    }


def psum_scatter_benchmark_pmap_calculate_metrics(
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
    trace_dir: str,
) -> Dict[str, Any]:
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
        trace_dir,
    )


def all_gather_benchmark_pmap(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str = None,
    sharding_strategy: str = None,
    op_dimension: int = 1,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the all_gather collective operation using pmap."""
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
    
    devices = jax.local_devices()[:ici_size]

    @functools.partial(jax.pmap, axis_name='i', devices=devices, in_axes=(None, 0))
    def jit_f(x, dummy):
        with jax.named_scope(MARKER):
            return jax.lax.all_gather(x, axis_name='i')

    m = matrix_dim
    n = BASE_SHAPE[1]
    k = BASE_SHAPE[2]

    def data_generator():
        matrix = jnp.ones((m, n, k), dtype=dtype)
        replicated_input = device_put_replicated(matrix, devices)
        dummy = device_put_sharded([jnp.array(i) for i in range(ici_size)], devices)
        return (replicated_input, dummy)

    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="all_gather_ici_op",
        trace_dir=trace_dir,
    )
    
    return {
        "ici_average_time_ms_list": time_ms_list,
        "matrix_shape": (m, n, k),
        "op_type": "AG",
        "trace_dir": trace_dir,
    }


def all_gather_benchmark_pmap_calculate_metrics(
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
    trace_dir: str,
) -> Dict[str, Any]:
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
        trace_dir,
    )


def all_to_all_benchmark_pmap(
    matrix_dim: int,
    dtype: jnp.dtype,
    ici_size: int,
    mesh_shape: str = None,
    sharding_strategy: str = None,
    op_dimension: int = 1,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the all_to_all collective operation using pmap."""
    libtpu_init_args = [
        "--xla_jf_debug_level=3",
        f"--xla_tpu_dvfs_p_state={GLOBAL_PSTATE}",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    
    devices = jax.local_devices()[:ici_size]

    @functools.partial(jax.pmap, axis_name='i', devices=devices, in_axes=(None, 0))
    def jit_f(x, dummy):
        with jax.named_scope(MARKER):
            return jax.lax.all_to_all(
                x, axis_name='i', split_axis=0, concat_axis=0, tiled=True
            )

    m = matrix_dim
    n = BASE_SHAPE[1]
    k = BASE_SHAPE[2]

    def data_generator():
        matrix = jnp.ones((m, n, k), dtype=dtype)
        replicated_input = device_put_replicated(matrix, devices)
        dummy = device_put_sharded([jnp.array(i) for i in range(ici_size)], devices)
        return (replicated_input, dummy)

    print("Running all_to_all pmap benchmark", num_runs, matrix_dim)
    time_ms_list = multiple_iteration_timeit_from_trace(
        jit_f,
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


def all_to_all_benchmark_pmap_calculate_metrics(
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
    trace_dir: str,
) -> Dict[str, Any]:
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
        trace_dir,
    )
