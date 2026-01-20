"""Benchmarks Host-to-Device and Device-to-Host transfer performance (Simple Baseline)."""

import time
import os
from typing import Any, Dict, Tuple, List

import jax
from jax import sharding
import numpy as np
from benchmark_utils import MetricsStatistics


libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
# 64 GiB
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "68719476736"
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "68719476736"

def get_tpu_devices(num_devices: int):
    devices = jax.devices()
    if len(devices) < num_devices:
        raise RuntimeError(f"Require {num_devices} devices, found {len(devices)}")
    return devices[:num_devices]

def benchmark_host_device(
    num_devices: int,
    data_size_mb: int,
    num_runs: int = 100,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks H2D/D2H transfer using simple device_put/device_get."""
    tpu_devices = get_tpu_devices(num_devices)
    
    num_elements = 1024 * 1024 * data_size_mb // np.dtype(np.float32).itemsize
    
    # Allocate Host Source Buffer
    host_data = np.random.normal(size=(num_elements,)).astype(np.float32)
    
    print(
        f"Benchmarking (Simple) Transfer with Data Size: {data_size_mb} MB on"
        f" {num_devices} devices for {num_runs} iterations"
    )

    # Setup Mesh Sharding (1D)
    mesh = sharding.Mesh(
        np.array(tpu_devices).reshape((num_devices,)), axis_names=("x",)
    )
    # Shard the 1D array across "x"
    partition_spec = sharding.PartitionSpec("x")
    
    data_sharding = sharding.NamedSharding(mesh, partition_spec)
    
    # Performance Lists
    h2d_perf, d2h_perf = [], []
        
    # Profiling Context
    import contextlib
    if trace_dir:
        profiler_context = jax.profiler.trace(trace_dir)
    else:
        profiler_context = contextlib.nullcontext()

    with profiler_context:
        # Warmup
        for _ in range(2):
            device_array = jax.device_put(host_data, data_sharding)
            device_array.block_until_ready()
            host_out = np.array(device_array)
            device_array.delete()
            del host_out

        for i in range(num_runs):
            # Step Context
            if trace_dir:
                step_context = jax.profiler.StepTraceAnnotation("host_device", step_num=i)
            else:
                step_context = contextlib.nullcontext()
            
            with step_context:
                 # H2D
                t0 = time.perf_counter()
                
                # Simple device_put
                device_array = jax.device_put(host_data, data_sharding)
                device_array.block_until_ready()
                
                t1 = time.perf_counter()
                h2d_perf.append((t1 - t0) * 1000)
                
                # Verify H2D shape/sharding
                assert device_array.shape == host_data.shape
                assert device_array.sharding == data_sharding
                
                # D2H
                t2 = time.perf_counter()
                
                # Simple device_get
                # Note: device_get returns a numpy array (copy)
                _ = jax.device_get(device_array)
                
                t3 = time.perf_counter()
                d2h_perf.append((t3 - t2) * 1000)
                
                device_array.delete()

    return {
        "H2D_Bandwidth_ms": h2d_perf,
        "D2H_Bandwidth_ms": d2h_perf,
    }

def benchmark_host_device_calculate_metrics(
    num_devices: int,
    data_size_mb: int,
    H2D_Bandwidth_ms: List[float],
    D2H_Bandwidth_ms: List[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculates metrics for Host-Device transfer."""
    params = locals().items()
    
    data_size_mib = data_size_mb
    
    # Filter out list params from metadata to avoid explosion
    metadata_keys = {
        "num_devices", 
        "data_size_mib", 
    }
    metadata = {k: v for k, v in params if k in metadata_keys}
    
    metrics = {}
    
    def add_metric(name, ms_list):
        # Report Bandwidth (GiB/s)
        # Handle division by zero if ms is 0
        bw_list = [
            ((data_size_mb / 1024) / (ms / 1000)) if ms > 0 else 0.0 
            for ms in ms_list
        ]
        stats_bw = MetricsStatistics(bw_list, f"{name}_bw (GiB/s)")
        metrics.update(stats_bw.serialize_statistics())

    add_metric("H2D", H2D_Bandwidth_ms)
    add_metric("D2H", D2H_Bandwidth_ms)

    return metadata, metrics
