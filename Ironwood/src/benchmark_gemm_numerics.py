"""
Benchmarks different compute operations in various flavors.
Considered ops:
1. gemm_fp8_rowwise
2. gemm_fp8_b128_fp32
3. gemm_fp8_rowwise_static_scaling
4. gemm_fp8_b128_fp32_static_scaling
5. gemm_mxfp8_b32
6. gemm_mxfp8_b32_static_scaling
"""

import os
from typing import Any, Dict, Callable


# pylint: disable=g-importing-member
from benchmark_utils import (
    iteration_timeit,
    ShardingStrategy,
    get_lhs_named_shading,
    get_rhs_named_shading,
    get_out_sharding,
    create_mesh,
    handle_based_on_sharding,
    unified_flops_metrics,
)
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from qwix import pallas as qpl
from qwix._src.core import qarray
from common import MARKER

# pylint: disable=g-importing-member
# Set the environment variable for TPU initialization arguments to optimize
# collective matmul. Setting the flags to false will disable the optimization.
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_enable_async_collective_fusion=true "
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    "--xla_tpu_overlap_compute_collective_tc=true "
    "--xla_enable_async_all_gather=true "
    "--xla_enable_async_collective_permute=true "
    "--xla_tpu_enable_all_experimental_scheduler_features=true "
    "--xla_tpu_accumulate_into_mrb=true "
    "--xla_tpu_scoped_vmem_limit_kib=65536 "
    "--xla_tpu_allow_conv_input_fusion_with_downcast_convert=true "
    "--xla_tpu_dvfs_p_state=7"
)

TRACE_BASE_DIR = None
METRICS_JSONL_DIR = None
# Matmul shapes: A(M,K) x B(K,N) = C(M,N)
M_STEP_SIZE = 1024
M_START_SIZE = 1024
M_MAX_SIZE = 50000
# The number of layers in the multilayer collective matmul.
# Matmul shapes: A(M,K) x H1(K,K)... x B(K,N) = C(M,N)
LAYERS = 2
WITH_SHARDING = True

SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
SEED = 0
PEAK_FLOPS_PER_DEVICE = 2307  # TFLOP/s for single core(device) of FP8


def gemm_fp8_quantization(
    m: int,
    k: int,
    n: int,
    f: Callable,
    num_runs: int = 1,
    trace_dir: str = None,
    task_name: str = "gemm_fp8_quantization",
) -> Dict[str, Any]:
    """FP8-Rowwise GEMM."""
    mesh = create_mesh(SHARDING_STRATEGY)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(lhs_sharding.spec, rhs_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)
    lhs_dtype = jnp.bfloat16
    rhs_dtype = jnp.bfloat16

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    # Run the benchmark
    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task=task_name,
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def gemm_fp8_rowwise(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8-Rowwise GEMM with dynamic scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[0],
            )
            qy = qpl.quantize(
                y,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[1],
            )
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    return gemm_fp8_quantization(
        m, k, n, f, num_runs, trace_dir, task_name="gemm_fp8_rowwise"
    )


def gemm_fp8_rowwise_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_fp8_rowwise_w_dequantize(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8-Rowwise GEMM with dynamic scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[0],
            )
            qy = qpl.quantize(
                y,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[1]
            )
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            ).astype(jnp.float32)
            final_result = acc * (
                qx.scale.astype(jnp.float32) * qy.scale.astype(jnp.float32)
            )
            return final_result.astype(jnp.bfloat16)

    mesh = create_mesh(SHARDING_STRATEGY)
    lhs_sharding = get_lhs_named_shading(mesh, SHARDING_STRATEGY)
    rhs_sharding = get_rhs_named_shading(mesh, SHARDING_STRATEGY)
    out_sharding = get_out_sharding(SHARDING_STRATEGY)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(lhs_sharding.spec, rhs_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)
    lhs_dtype = jnp.bfloat16
    rhs_dtype = jnp.bfloat16

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key  # Use and update the outer 'key'
        key, key_lhs, key_rhs = jax.random.split(key, 3)

        # Create random data on host
        lhs_host = jax.random.normal(key_lhs, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(key_rhs, rhs_shape).astype(rhs_dtype)

        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)

        return (lhs_device, rhs_device)

    # Run the benchmark
    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="gemm_fp8_rowwise_w_dequantize",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def gemm_fp8_rowwise_w_dequantize_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_fp8_b128_fp32(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8 GEMM as DeepSeek-stype quantization, block size: 1x128. Use dynamic scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[0],
                tiled_axes={1: 128},
            )
            qy = qpl.quantize(
                y,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="absmax",
                channelwise_axes=[1],
                tiled_axes={1: 128},
            )
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    return gemm_fp8_quantization(
        m, k, n, f, num_runs, trace_dir, task_name="gemm_fp8_b128_fp32"
    )


def gemm_fp8_b128_fp32_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_fp8_rowwise_static_scaling(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8-Rowwise GEMM with static scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="fixed, -224, 224",
                channelwise_axes=[0],
            )
            qy = qpl.quantize(
                y,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="fixed, -224, 224",
                channelwise_axes=[1],
            )
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    return gemm_fp8_quantization(
        m, k, n, f, num_runs, trace_dir, task_name="gemm_fp8_rowwise_static_scaling"
    )


def gemm_fp8_rowwise_static_scaling_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_fp8_b128_fp32_static_scaling(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8 GEMM as DeepSeek-stype quantization, block size: 1x128. Use static scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(
                x,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="fixed, -224, 224",
                channelwise_axes=[0],
                tiled_axes={1: 128},
            )
            qy = qpl.quantize(
                y,
                qtype=jnp.float8_e4m3fn,
                scale_dtype=jnp.float32,
                calibration_method="fixed, -224, 224",
                channelwise_axes=[1],
                tiled_axes={1: 128},
            )
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    return gemm_fp8_quantization(
        m, k, n, f, num_runs, trace_dir, task_name="gemm_fp8_b128_fp32_static_scaling"
    )


def gemm_fp8_b128_fp32_static_scaling_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_mxfp8_b32(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8-Rowwise GEMM with dynamic scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            how = qarray.HowToQuantize(qtype="mxfp8", calibration_method="absmax")
            qx = qarray.quantize(x, how=how)
            qy = qarray.quantize(y, how=how)
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    return gemm_fp8_quantization(
        m, k, n, f, num_runs, trace_dir, task_name="gemm_mxfp8_b32"
    )


def gemm_mxfp8_b32_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )


def gemm_mxfp8_b32_static_scaling(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """FP8-Rowwise GEMM with dynamic scaling factors."""

    def f(x, y):
        with jax.named_scope(MARKER):
            how = qarray.HowToQuantize(
                qtype="mxfp8", calibration_method="fixed, -224, 224"
            )
            qx = qarray.quantize(x, how=how)
            qy = qarray.quantize(y, how=how)
            acc = jax.numpy.einsum(
                "ij,jk->ik", qx.qvalue, qy.qvalue, preferred_element_type=jnp.float32
            )
            return acc.astype(jnp.bfloat16)

    return gemm_fp8_quantization(
        m, k, n, f, num_runs, trace_dir, task_name="gemm_mxfp8_b32"
    )


def gemm_mxfp8_b32_static_scaling_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops, total_flops_all_devices = handle_based_on_sharding(
        total_flops, SHARDING_STRATEGY
    )
    return unified_flops_metrics(
        m,
        n,
        k,
        time_ms_list,
        total_flops,
        total_flops_all_devices,
        PEAK_FLOPS_PER_DEVICE,
    )
