"""Benchmarks matmul in various flavors.

1. 
"""

import os
from typing import Any, Dict, Tuple


# pylint: disable=g-importing-member
from benchmark_utils import simple_timeit, MetricsStatistics, iteration_timeit
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np
from qwix import pallas as qpl
from flax import nnx
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
    "--xla_tpu_dvfs_p_state=7 "
    # "--xla_tpu_should_accumulate_into_mrb=true" # Unknown XLA Flag
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
SEED = 0
PEAK_FLOPS_PER_DEVICE=1153.5 # TFLOP/s for single core(device) under p_state=7

def create_mesh() -> Mesh:
    """Creates a mesh."""
    mesh = Mesh(np.array(jax.devices()), axis_names="i")
    return mesh

def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_param_keys = {"time_ms_list", "total_flops", "total_flops_all_devices", "peak_TFLOPS_per_device"}
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_param_keys
    }
    return metadata

def unified_flops_metrics(
    m: int, n: int, k: int, time_ms_list: list[float], total_flops: int, total_flops_all_devices: int, peak_TFLOPS_per_device: float
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_all_devices = [
        total_flops_all_devices / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    mfu = [
        tflops_per_sec/peak_TFLOPS_per_device for tflops_per_sec in tflops_per_sec_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec_pre_chip"
    )
    tflops_per_sec_all_devices_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_all_devices, metrics_name="tflops_per_sec"
    )
    mfu_statistics=MetricsStatistics(
        metrics_list=mfu, metrics_name="MFU"
    )
    print(
        f"Total floating-point ops: {total_flops}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, "
        f"Throughput (median): {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOP / second / chip, "
        f"TotalThroughput (median): {tflops_per_sec_all_devices_statistics.statistics['p50']:.2f} TFLOP / second, "
        f"MFU: {mfu_statistics.statistics['p50']:.2%}"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "StepTime(median,ms)": average_time_ms_statistics.statistics['p50'],
            "Throughput(median,TFLOP/s/chip)": tflops_per_sec_statistics.statistics['p50'],
            "TotalThroughput(median,TFLOP/s)": tflops_per_sec_all_devices_statistics.statistics['p50'],
            "MFU": mfu_statistics.statistics['p50'],
            "total_flops": total_flops,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(tflops_per_sec_all_devices_statistics.serialize_statistics())
    metrics.update(mfu_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics

def unified_bytes_metrics(
    time_ms_list: list[float], total_bytes: int
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    gigabytes_per_sec_list = [
        total_bytes / average_time_s / 10**9 for average_time_s in average_time_s_list
    ]
    average_time_ms_statistics = MetricsStatistics(
        metrics_list=time_ms_list, metrics_name="step_time_ms"
    )
    gigabytes_per_sec_statistics = MetricsStatistics(
        metrics_list=gigabytes_per_sec_list, metrics_name="tflops_per_sec"
    )
    print(
        f"Total bytes: {total_bytes}, Step Time (median): {average_time_ms_statistics.statistics['p50']:.2f}, Performance (median):"
        f" {gigabytes_per_sec_statistics.statistics['p50']:.2f} GBytes / second"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "StepTime(median,ms)": average_time_ms_statistics.statistics['p50'],
            "StepTime(average,ms)": average_time_ms_statistics.statistics['avg'],
            "StepTime(P90,ms)": average_time_ms_statistics.statistics['p90'],
            "Throughput(median,GBytes/s)": gigabytes_per_sec_statistics.statistics['p50'],
            "Throughput(average,GBytes/s)": gigabytes_per_sec_statistics.statistics['avg'],
            "Throughput(P90,GBytes/s)": gigabytes_per_sec_statistics.statistics['p90'],
            "total_bytes": total_bytes,
        }
    )
    metrics.update(average_time_ms_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics

def gemm_simple(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """Benchmarks the OUT<M, N>:BF16 = IN0<M, K>:FP8 x IN1<N, K>:FP8. Accumulation is FP32."""

    def f(x, y):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum("ij,jk->ik", x, y, preferred_element_type=jnp.float32)
            return acc.astype(jnp.bfloat16)

    mesh = create_mesh()
    rhs_sharding = NamedSharding(mesh, P(None, None))
    if WITH_SHARDING:
        lhs_sharding = NamedSharding(mesh, P("i", None))
        out_sharding = P("i", None)
    else:
        lhs_sharding = NamedSharding(mesh, P(None, None))
        out_sharding = P(None, None)

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
    lhs_dtype = jnp.float8_e4m3fn
    rhs_dtype = jnp.float8_e4m3fn

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key # Use and update the outer 'key'
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
        task="gemm_simple",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def gemm_simple_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops_all_devices = total_flops
    if WITH_SHARDING:
        total_flops = total_flops // jax.device_count()
    return unified_flops_metrics(m, n, k, time_ms_list, total_flops, total_flops_all_devices, PEAK_FLOPS_PER_DEVICE*2)

def gemm(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None
) -> Dict[str, Any]:
    """OUT<M, N>:BF16 = matmul(IN0<M, K>:FP8, IN1<N, K>:FP8) * outer_product(SF0<M, 1>:FP32 * SF1<1, N>:FP32)"""
    def f(x, y, scale_m, scale_n):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum("ij,jk->ik", x, y, preferred_element_type=jnp.float32)
            scales = scale_m * scale_n
            result_fp32 = acc * scales
            return result_fp32.astype(jnp.bfloat16)

    mesh = create_mesh()
    rhs_sharding = NamedSharding(mesh, P(None, None))
    sf1_sharding = NamedSharding(mesh, P(None, None))

    if WITH_SHARDING:
        lhs_sharding = NamedSharding(mesh, P("i", None))
        sf0_sharding = NamedSharding(mesh, P("i", None))
        out_sharding = P("i", None)
    else:
        lhs_sharding = NamedSharding(mesh, P(None, None))
        sf0_sharding = NamedSharding(mesh, P(None, None))
        out_sharding = P(None, None)

    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(lhs_sharding.spec, rhs_sharding.spec, sf0_sharding.spec, sf1_sharding.spec),
            out_specs=out_sharding,
            check_rep=False,
        )
    )

    lhs_shape = (m, k)
    rhs_shape = (k, n)
    sf0_shape = (m, 1)
    sf1_shape = (1, n)
    
    lhs_dtype = jnp.float8_e4m3fn
    rhs_dtype = jnp.float8_e4m3fn
    sf0_dtype = jnp.float32
    sf1_dtype = jnp.float32

    key = jax.random.key(SEED)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key # Use and update the outer 'key'
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        
        # Create random data on host
        lhs_host = jax.random.normal(k1, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(k2, rhs_shape).astype(rhs_dtype)
        sf0_host = jax.random.normal(k3, sf0_shape).astype(sf0_dtype)
        sf1_host = jax.random.normal(k4, sf1_shape).astype(sf1_dtype)
        
        # Put on device (HBM)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)
        sf0_device = jax.device_put(sf0_host, sf0_sharding)
        sf1_device = jax.device_put(sf1_host, sf1_sharding)
        
        return (lhs_device, rhs_device, sf0_device, sf1_device)
    
    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="gemm",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def gemm_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    total_flops_all_devices = total_flops
    if WITH_SHARDING:
        total_flops = total_flops // jax.device_count()
    return unified_flops_metrics(m, n, k, time_ms_list, total_flops, total_flops_all_devices, PEAK_FLOPS_PER_DEVICE*2)


def gemm_accum(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """OUT<M, N>:FP32 += matmul(IN0<M, K>:FP8, IN1<N, K>:FP8) * outer_product(SF0<M, 1>:FP32 * SF1<1, N>:FP32)"""
    def f(out_buffer, x, y, scale_m, scale_n):
        with jax.named_scope(MARKER):
            acc = jax.numpy.einsum("ij,jk->ik", x, y, preferred_element_type=jnp.float32)
            scales = scale_m * scale_n
            result_fp32 = acc * scales
            return out_buffer + result_fp32

    mesh = create_mesh()

    rhs_sharding = NamedSharding(mesh, P(None, None))
    sf1_sharding = NamedSharding(mesh, P(None, None))

    if WITH_SHARDING:
        lhs_sharding = NamedSharding(mesh, P("i", None))
        sf0_sharding = NamedSharding(mesh, P("i", None))
        out_buffer_sharding = NamedSharding(mesh, P("i", None))
        out_sharding = P("i", None)
    else:
        lhs_sharding = NamedSharding(mesh, P(None, None))
        sf0_sharding = NamedSharding(mesh, P(None, None))
        out_buffer_sharding = NamedSharding(mesh, P(None, None))
        out_sharding = P(None, None)
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(
                out_buffer_sharding.spec, 
                lhs_sharding.spec, 
                rhs_sharding.spec, 
                sf0_sharding.spec, 
                sf1_sharding.spec
            ),
            out_specs=out_sharding,
            check_rep=False,
        )
    )
    
    lhs_shape = (m, k)
    rhs_shape = (k, n)
    sf0_shape = (m, 1)
    sf1_shape = (1, n)
    out_buffer_shape = (m, n)
    
    lhs_dtype = jnp.float8_e4m3fn
    rhs_dtype = jnp.float8_e4m3fn
    sf0_dtype = jnp.float32
    sf1_dtype = jnp.float32
    out_buffer_dtype = jnp.float32
    
    key = jax.random.key(0)

    def data_generator():
        """Creates new random data on host and puts it on device."""
        nonlocal key # Use and update the outer 'key'
        key, k_buf, k1, k2, k3, k4 = jax.random.split(key, 6)
        
        # Create random data on host
        out_buffer_host = jax.random.normal(k_buf, out_buffer_shape).astype(out_buffer_dtype)
        lhs_host = jax.random.normal(k1, lhs_shape).astype(lhs_dtype)
        rhs_host = jax.random.normal(k2, rhs_shape).astype(rhs_dtype)
        sf0_host = jax.random.normal(k3, sf0_shape).astype(sf0_dtype)
        sf1_host = jax.random.normal(k4, sf1_shape).astype(sf1_dtype)
        
        # Put on device (HBM)
        out_buffer_device = jax.device_put(out_buffer_host, out_buffer_sharding)
        lhs_device = jax.device_put(lhs_host, lhs_sharding)
        rhs_device = jax.device_put(rhs_host, rhs_sharding)
        sf0_device = jax.device_put(sf0_host, sf0_sharding)
        sf1_device = jax.device_put(sf1_host, sf1_sharding)
        
        return (out_buffer_device, lhs_device, rhs_device, sf0_device, sf1_device)

    time_ms_list = iteration_timeit(
        jit_sharded_f,
        data_generator,
        matrix_dim=f"{m}x{n}x{k}",
        tries=num_runs,
        task="gemm_accum",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def gemm_accum_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    # Calculate FLOPs
    total_flops = 2 * m * k * n + m * n  # Total floating-point operations
    total_flops_all_devices = total_flops
    if WITH_SHARDING:
        total_flops = total_flops // jax.device_count()
    return unified_flops_metrics(m, n, k, time_ms_list, total_flops, total_flops_all_devices, PEAK_FLOPS_PER_DEVICE*2)


def quantization(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    OUT<M, N>:FP8, SF<M>:FP32 = Quantize(N<M, N>:BF16)
    SF[i] = FP8_MAX / amax(IN[i])
    OUT[i] = cast_fp8(IN[i] / SF[i])
    """
    def f(x):
        with jax.named_scope(MARKER):
            qx = qpl.quantize(x, qtype=jnp.float8_e4m3fn, scale_dtype=jnp.float32, calibration_method="absmax", channelwise_axes=[0])
            return qx.qvalue, qx.scale

    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=P("i", None),
            out_specs=(P("i", None), P("i", None)),
            check_rep=False,
        )
    )
    # Run once.
    output, scale = jit_sharded_f(x)
    jax.block_until_ready((output, scale))  # Ensure full completion before printing metrics
    print(f"qwix({x.shape=}) = {output.shape=} x {scale.shape=}, {x.dtype=}, {output.dtype=}, {scale.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jit_sharded_f,
        x,
        tries=num_runs,
        task="quantization",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def quantization_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 5 * m * n + 4 * m  # Total floating-point operations
    return unified_bytes_metrics(time_ms_list, total_bytes)

def transpose_quantization(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    OUT<N, M>:FP8, SF<N>:FP32 = Quantize(Transpose(N<M, N>:BF16)) for 2D
    SF[i] = FP8_MAX / amax(IN[i])
    OUT[i] = cast_fp8(IN[i] / SF[i])
    """
    def f(x):
        with jax.named_scope(MARKER):
            x = x.T
            qx = qpl.quantize(x, qtype=jnp.float8_e4m3fn, scale_dtype=jnp.float32, calibration_method="absmax", channelwise_axes=[0])
            return qx.qvalue, qx.scale

    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=P("i", None),
            out_specs=(P(), P()),
            check_rep=False,
        )
    )
    # Run once.
    output, scale = jit_sharded_f(x)
    jax.block_until_ready((output, scale))  # Ensure full completion before printing metrics
    print(f"qwix({x.shape=}) = {output.shape=} x {scale.shape=}, {x.dtype=}, {output.dtype=}, {scale.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jit_sharded_f,
        x,
        tries=num_runs,
        task="quantization",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def transpose_quantization_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 5 * m * n + 4 * m  # Total floating-point operations
    return unified_bytes_metrics(time_ms_list, total_bytes)

def swiglu_fwd(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    A, B = Split(X, 2)
    Y = Swish(A) ⊗ B
    """
    def f(x):
        with jax.named_scope(MARKER):
            A, B = jnp.split(x, 2, axis=-1)
            A_fp32 = A.astype(jnp.float32)
            B_fp32 = B.astype(jnp.float32)
            Y_fp32 = jax.nn.silu(A_fp32) * B_fp32
            return Y_fp32.astype(jnp.bfloat16)


    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=P("i", None),
            out_specs=(P("i", None)),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(x)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"swiglu_fwd({x.shape=}) = {output.shape=}, {x.dtype=}, {output.dtype=}")
    time_ms_list = simple_timeit(
        jit_sharded_f,
        x,
        tries=num_runs,
        task="swiglu_fwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def swiglu_fwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (m * n + m * n // 2)
    return unified_bytes_metrics(time_ms_list, total_bytes)


def swiglu_bwd(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    Inverse of swiglu_fwd
    """
    def f_fwd(x):
        with jax.named_scope(MARKER):
            A, B = jnp.split(x, 2, axis=-1)
            A_fp32 = A.astype(jnp.float32)
            B_fp32 = B.astype(jnp.float32)
            Y_fp32 = jax.nn.silu(A_fp32) * B_fp32
            return Y_fp32.astype(jnp.bfloat16)
    
    def f_bwd(x: jax.Array, dy: jax.Array) -> jax.Array:
        """
        x: The original <M, N> BF16 input.
        dy: The upstream <M, N/2> BF16 gradient.
        """
        with jax.named_scope(MARKER):
            # Get the VJP "pullback" function
            # We ignore the forward result (_y)
            _y, pullback_fn = jax.vjp(f_fwd, x)
            
            # Call the pullback function with the upstream gradient
            # This IS the backward pass.
            dx = pullback_fn(dy)
            
            # dx is returned as a tuple (one item per arg of f_fwd)
            return dx[0]

    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    dy = jnp.arange(np.prod((m, n // 2))).reshape((m, n // 2)).astype(jnp.bfloat16)
    dy = jax.device_put(dy, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f_bwd,
            mesh,
            in_specs=(P("i", None), P("i", None)),
            out_specs=(P("i", None)),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(x, dy)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"swiglu_bwd({x.shape=}, {dy.shape=}) = {output.shape=}, {x.dtype=}, {dy.dtype=}, {output.dtype=}")
    time_ms_list = simple_timeit(
        jit_sharded_f,
        x,
        dy,
        tries=num_runs,
        task="swiglu_bwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def swiglu_bwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (2 * m * n + m * n // 2)
    return unified_bytes_metrics(time_ms_list, total_bytes)

def rmsnorm_fwd(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    For each row i of N:
    Y_i = X_i / rms(x_i)
    """
    with jax.named_scope(MARKER):
        f = nnx.RMSNorm(num_features=n, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))

    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=P("i", None),
            out_specs=(P("i", None)),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(x)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"rmsnorm_fwd({x.shape=}) = {output.shape=}, {x.dtype=}, {output.dtype=}")
    time_ms_list = simple_timeit(
        jit_sharded_f,
        x,
        tries=num_runs,
        task="rmsnorm_fwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def rmsnorm_fwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (2 * m * n + m * n)
    return unified_bytes_metrics(time_ms_list, total_bytes)

def rmsnorm_bwd(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    Inverse of rmsnorm_fwd
    """
    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    f = nnx.RMSNorm(num_features=n, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    # We need a scalar loss function to differentiate.
    # We sum the output and cast to f32 for stable gradients.
    def loss_fn(module: nnx.RMSNorm, x_input: jax.Array):
        with jax.named_scope(MARKER):
            y = module(x_input)
        local_loss = jnp.sum(y.astype(jnp.float32))
        return jax.lax.psum(local_loss, axis_name='i')

    sharded_loss_fn = shard_map(
        loss_fn,
        mesh,
        in_specs=(P(), P("i", None)),
        out_specs=P(), # Output is a single replicated scalar
        check_rep=False
    )
    grad_fn = nnx.grad(sharded_loss_fn, argnums=1)
    jit_sharded_bwd = jax.jit(grad_fn)

    # Run once.
    grads = jit_sharded_bwd(f, x)
    jax.block_until_ready(grads)  # Ensure full completion before printing metrics
    print(f"rmsnorm_bwd({x.shape=}) = {grads.shape=}, {x.dtype=}, {grads.dtype=}")
    time_ms_list = simple_timeit(
        jit_sharded_bwd,
        f,
        x,
        tries=num_runs,
        task="rmsnorm_bwd",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def rmsnorm_bwd_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 2 * (2 * m * n + m * n)
    return unified_bytes_metrics(time_ms_list, total_bytes)

def add(m: int, n: int, num_runs: int = 1, trace_dir: str = None, 
) -> Dict[str, Any]:
    """
    Z = X + Y
    """
    def f(x, y):
        with jax.named_scope(MARKER):
            return x + y


    mesh = create_mesh()
    x = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    x = jax.device_put(x, NamedSharding(mesh, P("i", None)))
    y = jnp.arange(np.prod((m, n))).reshape((m, n)).astype(jnp.bfloat16)
    y = jax.device_put(y, NamedSharding(mesh, P("i", None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P("i", None), P("i", None)),
            out_specs=(P("i", None)),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(x, y)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"add({x.shape=}, {y.shape=}) = {output.shape=}, {x.dtype=}, {y.dtype=}, {output.dtype=}")
    time_ms_list = simple_timeit(
        jit_sharded_f,
        x,
        y,
        tries=num_runs,
        task="add",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}

def add_calculate_metrics(
    m: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    total_bytes = 6 * m * n
    return unified_bytes_metrics(time_ms_list, total_bytes)