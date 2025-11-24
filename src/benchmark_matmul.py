"""Benchmarks matmul in various flavors.

1. naive_matmul: matmul using jax.numpy.einsum. This is the baseline.
2. multilayer_collective_matmul: Collective matmul with multiple layers. With
async collective enabled, the communication for next layer overlaps with the
compute of previous layer.
3. collective_matmul_one_direction: Collective matmul that overlaps the permute
with the compute. The permute is done in one direction.
4. collective_matmul_two_directions: Collective matmul that overlaps the permute
with the compute. The permute is done in two directions.
"""

import os
from typing import Any, Dict, Tuple


# pylint: disable=g-importing-member
from benchmark_utils import simple_timeit, MetricsStatistics
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

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
    "--xla_tpu_enable_all_experimental_scheduler_features=true"
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


def create_mesh() -> Mesh:
    """Creates a mesh."""
    mesh = Mesh(np.array(jax.devices()), axis_names="i")
    return mesh


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_param_keys = {"time_ms_list"}
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_param_keys
    }
    return metadata


def naive_matmul(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the jax.numpy.einsum."""

    def f(x, y):
        return jax.numpy.einsum("ij,jk->ik", x, y)

    mesh = create_mesh()
    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.bfloat16)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.bfloat16)
    # lhs(m,k): sharded across devices. rhs(k,n): replicated on devices.
    # output(m,n): replicated on devices.
    lhs = jax.device_put(lhs, NamedSharding(mesh, P("i", None)))
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P(), P()),
            out_specs=P(),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(lhs, rhs)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"{lhs.shape=} x {rhs.shape=} = {output.shape=}, {output.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jit_sharded_f,
        lhs,
        rhs,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="naive_matmul",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def naive_matmul_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    total_gigabytes_transferred = 2 * (m * k + k * n + m * n) / 10**9
    data_transfer_gbyte_sec_list = [
        total_gigabytes_transferred / average_time_s
        for average_time_s in average_time_s_list
    ]
    data_transfer_gbyte_sec_statistics = MetricsStatistics(
        metrics_list=data_transfer_gbyte_sec_list,
        metrics_name="data_transfer_gbyte_sec",
    )
    print(
        f"Total floating-point ops: {total_flops}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOPs / second, Total GBs transferred (median):"
        f" {total_gigabytes_transferred:.2f} GB, GBs per second:"
        f" {data_transfer_gbyte_sec_statistics.statistics['p50']:.2f} GB/s"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "total_flops": total_flops,
            "total_gigabytes_transferred": total_gigabytes_transferred,
        }
    )
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(data_transfer_gbyte_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def single_host_naive_matmul(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks matmul on a single device without any sharding."""

    def f(x, y):
        return jax.numpy.einsum("ij,jk->ik", x, y)

    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.bfloat16)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.bfloat16)
    # Put both matrices on device 0 without any sharding
    jitted_f = jax.jit(f)
    # Run once
    output = jitted_f(lhs, rhs)
    jax.block_until_ready(output)
    print(f"{lhs.shape=} x {rhs.shape=} = {output.shape=}, {output.dtype=}")
    # Run the benchmark
    time_ms_list = simple_timeit(
        jitted_f,
        lhs,
        rhs,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="single_host_naive_matmul",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def single_host_naive_matmul_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the single host naive matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    total_gigabytes_transferred = 2 * (m * k + k * n + m * n) / 10**9
    data_transfer_gbyte_sec_list = [
        total_gigabytes_transferred / average_time_s
        for average_time_s in average_time_s_list
    ]
    data_transfer_gbyte_sec_statistics = MetricsStatistics(
        metrics_list=data_transfer_gbyte_sec_list,
        metrics_name="data_transfer_gbyte_sec",
    )
    print(
        f"Total floating-point ops: {total_flops}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50']:.2f} TFLOPs / second, Total GBs transferred (median):"
        f" {total_gigabytes_transferred:.2f} GB, GBs per second:"
        f" {data_transfer_gbyte_sec_statistics.statistics['p50']:.2f} GB/s"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "total_flops": total_flops,
            "total_gigabytes_transferred": total_gigabytes_transferred,
        }
    )
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics.update(data_transfer_gbyte_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def collective_matmul_one_direction(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the collective matmul that does permute in one direction."""

    def f(lhs, rhs):
        # lhs is the looped operand; rhs is the local operand
        axis_size = jax.lax.psum(1, axis_name="i")
        axis_index = jax.lax.axis_index(axis_name="i")
        chunk_size = lhs.shape[0]

        def scanned_call(i, carrys):
            accum, lhs = carrys
            # matmul for a chunk
            update = lhs @ rhs
            # circular shift to the left
            lhs = jax.lax.ppermute(
                lhs,
                axis_name="i",
                perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
            )
            # device 0 computes chunks 0, 1, ...
            # device 1 computes chunks 1, 2, ...
            update_index = (((axis_index + i) % axis_size) * chunk_size, 0)
            accum = jax.lax.dynamic_update_slice(accum, update, update_index)
            return accum, lhs

        accum = jnp.zeros((lhs.shape[0] * axis_size, rhs.shape[1]), dtype=lhs.dtype)
        for i in range(0, axis_size - 1):
            accum, lhs = scanned_call(i, (accum, lhs))
        # compute the last chunk, without the ppermute
        update = lhs @ rhs
        i = axis_size - 1
        update_index = (((axis_index + i) % axis_size) * chunk_size, 0)
        accum = jax.lax.dynamic_update_slice(accum, update, update_index)
        return accum

    mesh = create_mesh()
    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.bfloat16)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.bfloat16)
    lhs = jax.device_put(lhs, NamedSharding(mesh, P("i", None)))
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P("i", None), P(None)),
            out_specs=P(None),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(lhs, rhs)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"{lhs.shape=} x {rhs.shape=} = {output.shape=}, {output.dtype=}")
    time_ms_list = simple_timeit(
        jit_sharded_f,
        lhs,
        rhs,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="collective_matmul_one_direction",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def collective_matmul_one_direction_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the collective matmul one direction benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    print(
        f"Total floating-point ops: {total_flops}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50'] :.2f} TFLOPs / second"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "total_flops": total_flops,
        }
    )
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def collective_matmul_two_directions(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the collective matmul that does permute in two directions."""

    def f(activations, weights):
        """Collective matrix multiply."""
        axis_size = jax.lax.psum(1, axis_name="i")
        axis_index = jax.lax.axis_index(axis_name="i")
        # Current sequence chunk
        chunk_size = activations.shape[1]
        mid_chunk = chunk_size // 2
        # Create accumulation buffer across all devices
        accum = jnp.zeros(
            (activations.shape[0] * axis_size, weights.shape[1]),
            dtype=activations.dtype,
        )
        # Compute and place initial chunk result in accum
        update = activations @ weights
        update_index = (axis_index * chunk_size, 0)
        accum = jax.lax.dynamic_update_slice(accum, update, update_index)
        # Prepare forward and backward activations for next steps
        activation_forward, activation_backward = jnp.split(activations, 2, axis=0)
        # Initial ppermute of activations to the next device
        activation_forward = jax.lax.ppermute(
            activation_forward,
            axis_name="i",
            perm=[(j, (j + 1) % axis_size) for j in range(axis_size)],
        )
        activation_backward = jax.lax.ppermute(
            activation_backward,
            axis_name="i",
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
        )

        # Define scanning function to handle chunked activation passing
        def scanned_call(i, carrys):
            accum, activation_forward, activation_backward = carrys
            # Forward and backward updates for each activation chunk
            update_forward = activation_forward @ weights
            update_backward = activation_backward @ weights
            # Propagate activations
            activation_forward = jax.lax.ppermute(
                activation_forward,
                axis_name="i",
                perm=[(j, (j + 1) % axis_size) for j in range(axis_size)],
            )
            activation_backward = jax.lax.ppermute(
                activation_backward,
                axis_name="i",
                perm=[(j, (j - 1) % axis_size) for j in range(axis_size)],
            )
            # Update indices for forward and backward propagation
            forward_update_index = ((axis_index - i - 1) % axis_size) * chunk_size
            backward_update_index = (
                (axis_index + i + 1) % axis_size
            ) * chunk_size + mid_chunk
            # Update accum with the calculated forward and backward updates
            accum = jax.lax.dynamic_update_slice(
                accum, update_forward, (forward_update_index, 0)
            )
            accum = jax.lax.dynamic_update_slice(
                accum, update_backward, (backward_update_index, 0)
            )
            return accum, activation_forward, activation_backward

        # Execute loop to propagate all chunks and collect results
        for i in range(0, axis_size - 1):
            accum, activation_forward, activation_backward = scanned_call(
                i, (accum, activation_forward, activation_backward)
            )
        return accum

    mesh = create_mesh()
    lhs = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.bfloat16)
    rhs = jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.bfloat16)
    lhs = jax.device_put(lhs, NamedSharding(mesh, P("i", None)))
    rhs = jax.device_put(rhs, NamedSharding(mesh, P(None, None)))
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P("i", None), P(None)),
            out_specs=P(None),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(lhs, rhs)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"{lhs.shape=} x {rhs.shape=} = {output.shape=}, {output.dtype=}")
    # Run the benchmark.
    time_ms_list = simple_timeit(
        jit_sharded_f,
        lhs,
        rhs,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="collective_matmul_two_directions",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def collective_matmul_two_directions_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the collective matmul two direction benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    total_flops = 2 * m * k * n  # Total floating-point operations
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    print(
        f"Total floating-point ops: {total_flops}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50'] :.2f} TFLOPs / second"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "total_flops": total_flops,
        }
    )
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def multilayer_collective_matmul(
    m: int, k: int, n: int, num_runs: int = 1, trace_dir: str = None, warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the multilayer collective matmul."""

    def f(act, weights):
        for weight in weights:
            act = act @ weight
        return act

    mesh = create_mesh()
    activation = jnp.arange(np.prod((m, k))).reshape((m, k)).astype(jnp.bfloat16)
    hidden_layers = [
        jnp.arange(np.prod((k, k))).reshape((k, k)).astype(jnp.bfloat16)
        for _ in range(LAYERS - 1)
    ]
    last_layer = [jnp.arange(np.prod((k, n))).reshape((k, n)).astype(jnp.bfloat16)]
    weights = hidden_layers + last_layer
    activation_sharding = NamedSharding(mesh, P("i", None))
    weight_sharding = NamedSharding(mesh, P(None, "i"))
    activation = jax.device_put(activation, activation_sharding)
    weights = [jax.device_put(weight, weight_sharding) for weight in weights]
    jit_sharded_f = jax.jit(
        shard_map(
            f,
            mesh,
            in_specs=(P(), P(None)),
            out_specs=P(None),
            check_rep=False,
        )
    )
    # Run once.
    output = jit_sharded_f(activation, weights)
    jax.block_until_ready(output)  # Ensure full completion before printing metrics
    print(f"Activation shape: {activation.shape}")
    print("Weights shapes:", [w.shape for w in weights])
    print(f"Output shape: {output.shape}, Output dtype: {output.dtype}")
    time_ms_list = simple_timeit(
        jit_sharded_f,
        activation,
        weights,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="collective_multilayer_matmul",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list}


def multilayer_collective_matmul_calculate_metrics(
    m: int, k: int, n: int, time_ms_list: list[float]
) -> Dict[str, Any]:
    """Calculates the metrics for the multilayer collective matmul benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata = get_metrics_helper(params)
    metrics = {}

    # Calculate FLOPs
    per_layer_flops = 2 * m * k * k  # Total floating-point operations
    last_layer_flops = 2 * m * k * n
    total_flops = per_layer_flops * (LAYERS - 1) + last_layer_flops
    average_time_s_list = [average_time_ms / 10**3 for average_time_ms in time_ms_list]
    tflops_per_sec_list = [
        total_flops / average_time_s / 10**12 for average_time_s in average_time_s_list
    ]
    tflops_per_sec_statistics = MetricsStatistics(
        metrics_list=tflops_per_sec_list, metrics_name="tflops_per_sec"
    )
    print(
        f"Total floating-point ops: {total_flops}, Performance (median):"
        f" {tflops_per_sec_statistics.statistics['p50'] :.2f} TFLOPs / second"
    )
    print()
    # Gather the metrics to report.
    metadata.update(
        {
            "total_flops": total_flops,
        }
    )
    metrics.update(tflops_per_sec_statistics.serialize_statistics())
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
