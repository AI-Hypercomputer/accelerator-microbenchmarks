"""Benchmarking p2p source target transfer."""

import os
from typing import Any, Dict, List, Tuple
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.sharding
from benchmark_utils import (
    MetricsStatistics,
    get_trace,
)
from common import MARKER
import tempfile

P = jax.sharding.PartitionSpec

os.environ['LIBTPU_INIT_ARGS'] = (
    '--xla_tpu_collect_sflag_wait_stats_trace=true '
    '--xla_tpu_force_global_barriers=true '
    '--xla_tpu_ragged_all_to_all_max_rdma_size_kib=-1 '
    '--xla_tpu_dvfs_p_state=7 '
)


def _run_under_xprof(
    function: jax.stages.Compiled,
    inputs: list[jax.Array],
    n_repeats: int,
    task: str,
):
  """Runs a function under xprof."""
  # warmup
  jax.block_until_ready(function(*inputs))
  with tempfile.TemporaryDirectory() as tmp_trace_dir:
    with jax.profiler.trace(tmp_trace_dir, create_perfetto_link=False):
      for i in range(n_repeats):
        with jax.profiler.StepTraceAnnotation(task, step_num=i):
          with jax.named_scope(f"{MARKER}_{i}"):
            result = function(*inputs)
            jax.block_until_ready(result)
    jtrace = get_trace(tmp_trace_dir)

    marker_done_events = []
    for event in jtrace["traceEvents"]:
        args = event.get("args", {})
        tf_op = args.get("tf_op", "")
        if MARKER in tf_op:
            marker_done_events.append(event)
    # when offloaded to sparse core look for call-done events
    marker_call_done_events = [
        e for e in marker_done_events if e.get("name", "").endswith("call-done")
    ]
    if marker_call_done_events:
        marker_done_events = marker_call_done_events
    durations_ms = [
        float(e["args"]["device_duration_ps"]) / 1e9 for e in marker_done_events
    ]
    return max(durations_ms)


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_keys = {'time_ms_list'}
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_keys
    }
    metadata['dtype'] = metadata['dtype'].dtype.itemsize
    return metadata


def send_recv_benchmark(
    source_id: int,
    target_id: int,
    num_elements: int,
    n_repeats: int,
    dtype: jnp.dtype,
    trace_dir: str,
):
    """Runs p2p communication, sending tensor_size_bytes from source to target device."""
    device_count = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((device_count,))
    mesh = jax.sharding.Mesh(devices, 'x')
    item_size = jnp.dtype(dtype).itemsize
    tensor_size_bytes = num_elements * item_size
    last_dim = tensor_size_bytes // (1 * 8 * item_size)

    def p2p_send(source_id, target_id):
        # Get the ID of the current device this code is running on
        device_id = jax.lax.axis_index('x')
        axis_size = jax.lax.axis_size('x')
        input_offsets = jnp.zeros((axis_size,), dtype=jnp.int32)
        output_offsets = jnp.zeros((axis_size,), dtype=jnp.int32)
        no_sends = jnp.zeros((axis_size,), dtype=jnp.int32)
        no_recvs = jnp.zeros((axis_size,), dtype=jnp.int32)

        # Only device `source_id` sends, and it sends to `target_id`.
        sender_send_sizes = jax.nn.one_hot(target_id, axis_size, dtype=jnp.int32)
        # Only device `target_id` receives, and it receives from `source_id`.
        target_recv_sizes = jax.nn.one_hot(source_id, axis_size, dtype=jnp.int32)

        final_send_sizes = jax.lax.select(
            device_id == source_id,
            sender_send_sizes,
            no_sends,
        )
        final_recv_sizes = jax.lax.select(
            device_id == target_id,
            target_recv_sizes,
            no_recvs,
        )
        input = jax.random.normal(jax.random.key(0), (1, 8, last_dim), dtype=dtype)
        output = jnp.zeros((1, 8, last_dim), dtype=dtype)

        with jax.named_scope(MARKER):
            ra2a = jax.lax.ragged_all_to_all(
                operand=input,
                output=output,
                input_offsets=input_offsets,
                send_sizes=final_send_sizes,
                output_offsets=output_offsets,
                recv_sizes=final_recv_sizes,
                axis_name='x',
            )
        max_val = jax.lax.reduce_max(ra2a, axes=(0, 1, 2))
        return max_val

    compiled_function = (
        jax.jit(
            jax.shard_map(
                p2p_send, mesh=mesh, out_specs=P(), in_specs=(P(), P())
            ),
            static_argnums=(0, 1),
        )
        .lower(source_id, target_id)
        .compile()
    )

  # Measures the longest wait time in milliseconds, across all the runs.
    runtime_ms = _run_under_xprof(
        compiled_function, [], n_repeats, f'p2p_{source_id}_to_{target_id}'
    )

    return {'runtime_ms': runtime_ms}


def send_recv_benchmark_calculate_metrics(
    source_id: int,
    target_id: int,
    num_elements: int,
    n_repeats: int,
    dtype: jnp.dtype,
    runtime_ms: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculates metrics for p2p benchmark."""
    params = locals().items()

    metadata = get_metrics_helper(params)
    metrics = {}

    tensor_size_bytes = num_elements * jnp.dtype(dtype).itemsize
    tensor_size_gbytes = tensor_size_bytes / 10**9

    metrics['runtime_ms (ms)'] = runtime_ms
    runtime_s = runtime_ms / 10**3
    metrics['achieved_bw (GB/s)'] = tensor_size_gbytes / runtime_s

    # Gather the metrics to report.
    metadata.update({
        'tensor_size_gbytes': tensor_size_gbytes,
    })

    metrics = {key: value for key, value in metrics.items() if value is not None}
    print(metadata)
    print(metrics)
    return metadata, metrics
