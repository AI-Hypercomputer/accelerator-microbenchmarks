"""Benchmarks Host-to-Device and Device-to-Host transfer performance."""

import concurrent.futures
import contextlib
import gc
import time
import os
from typing import Any, Dict, Tuple, List, Optional

import jax
from jax import sharding
import numpy as np
from benchmark_utils import MetricsStatistics

libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "68719476736" # 64 GiB
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "68719476736"

def get_tpu_devices(num_devices: int):
    devices = jax.devices()
    if len(devices) < num_devices:
        raise RuntimeError(f"Require {num_devices} devices, found {len(devices)}")
    return devices[:num_devices]

# --- Smart Chunking Implemention Helpers ---

def _run_h2d_chunked(host_shards, target_devices, num_devices, chunks_per_device):
    chk_h2d_start = time.perf_counter()
    total_workers = num_devices * chunks_per_device
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
        chunked_futures = []
        for shard, dev in zip(host_shards, target_devices):
            sub_chunks = np.array_split(shard, chunks_per_device, axis=0)
            for chunk in sub_chunks:
                chunked_futures.append(
                    executor.submit(jax.device_put, chunk, dev)
                )
        chunked_buffers = [f.result() for f in chunked_futures]
        for db in chunked_buffers:
            db.block_until_ready()
    chk_h2d_end = time.perf_counter()
    h2d_ms = (chk_h2d_end - chk_h2d_start) * 1000
    for db in chunked_buffers:
        db.delete()
    return h2d_ms

def _run_d2h_chunked(host_data, data_sharding, num_devices, chunks_per_device):
    data_on_device = jax.device_put(host_data, data_sharding)
    data_on_device.block_until_ready()
    
    total_workers = num_devices * chunks_per_device
    chk_d2h_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
        d2h_futures = []
        for shard in data_on_device.addressable_shards:
            # Direct slicing on device array to avoid copy
            shard_len = shard.data.shape[0]
            chunk_size = (shard_len + chunks_per_device - 1) // chunks_per_device
            for i in range(chunks_per_device):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, shard_len)
                if start < end:
                    d2h_futures.append(
                        executor.submit(jax.device_get, shard.data[start:end])
                    )
        _ = [f.result() for f in d2h_futures]
    chk_d2h_end = time.perf_counter()
    d2h_ms = (chk_d2h_end - chk_d2h_start) * 1000
    data_on_device.delete()
    return d2h_ms

def _run_chunked(host_data, data_sharding, host_shards, target_devices, num_devices, chunks_per_device_h2d, chunks_per_device_d2h):
    h2d_ms = _run_h2d_chunked(host_shards, target_devices, num_devices, chunks_per_device_h2d)
    d2h_ms = _run_d2h_chunked(host_data, data_sharding, num_devices, chunks_per_device_d2h)
    return h2d_ms, d2h_ms

def _run_warmup(host_data, data_sharding, data_size_mb):
    # --- ADAPTIVE WARM UP ---
    if data_size_mb <= 128:
        warmup_iters = 50
    elif data_size_mb >= 8192:
        warmup_iters = 3
    else:
        warmup_iters = 10

    for _ in range(warmup_iters):
        data_on_device = jax.device_put(host_data, data_sharding)
        data_on_device.block_until_ready()
        _ = jax.device_get(data_on_device)
        data_on_device.delete()

    gc.collect()

def _get_chunks_per_device(data_size_mb, num_devices):
    # --- SMART CHUNKING CONFIG ---
    target_chunk_size_mb = 16
    max_global_threads = 256
    
    data_per_device_mb = data_size_mb / num_devices

    if data_per_device_mb < target_chunk_size_mb:
        chunks_per_device = 1
    else:
        chunks_per_device = int(data_per_device_mb / target_chunk_size_mb)

    total_threads = num_devices * chunks_per_device
    if total_threads > max_global_threads:
        chunks_per_device = max(1, int(max_global_threads / num_devices))
    
    return chunks_per_device


def _find_optimal_chunk_size(
    run_fn,
    num_devices,
    data_size_mb,
    search_min_size_mb=1,
    max_global_threads=256
):
    """Finds optimal chunk size by iterating over candidates."""
    print("  Searching for optimal chunk size...")
    
    # Generate size candidates
    candidates_mb = []
    curr = search_min_size_mb
    data_per_device_mb = data_size_mb / num_devices
    
    # Iterate until we cover the full data size per device
    while curr <= data_per_device_mb:
        candidates_mb.append(curr)
        curr *= 2
    # Ensure we test at least one candidate (e.g. if data < min_size)
    if not candidates_mb:
         candidates_mb.append(data_per_device_mb)

    # Map sizes to counts, keeping track of unique counts to test
    candidates_counts = []
    seen_counts = set()
    
    for size_mb in candidates_mb:
        if size_mb > data_per_device_mb:
             count = 1
        else:
             count = int(data_per_device_mb / size_mb)
             if count < 1: count = 1
             
        # Filter by max global threads
        if (count * num_devices) > max_global_threads:
            continue
        
        if count not in seen_counts:
            candidates_counts.append(count)
            seen_counts.add(count)
            
    # Sort candidates (counts) ascending for clean output
    candidates_counts.sort()
    
    if not candidates_counts:
        candidates_counts = [1]

    best_chunk_count = 1
    best_median_bw = -1.0
    
    # 5 search iterations + 3 warmup (before search)
    warmup_iters = 3
    search_iters = 5
    
    try:
        for _ in range(warmup_iters):
             run_fn(1) # Warmup with 1 chunk
    except Exception:
        pass 
        
    for chunk_count in candidates_counts:
        times_ms = []
        try:
            for _ in range(search_iters):
                t_start = time.perf_counter()
                res = run_fn(chunk_count)
                t_end = time.perf_counter()
                
                if isinstance(res, (int, float)):
                    times_ms.append(res)
                else:
                    times_ms.append((t_end - t_start) * 1000)
            
            median_ms = np.median(times_ms)
            if median_ms > 0:
                 if best_median_bw < 0 or median_ms < best_median_bw:
                     best_median_bw = median_ms
                     best_chunk_count = chunk_count
        except Exception as e:
            continue
            
    print(f"  Found optimal chunk count: {best_chunk_count} (approx size: {data_per_device_mb/best_chunk_count:.2f} MB)")
    return best_chunk_count


def benchmark_host_device_smart_chunking(
    num_devices: int,
    data_size_mb: int,
    num_runs: int = 100,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks H2D/D2H transfer using smart chunking."""
    tpu_devices = get_tpu_devices(num_devices)
    
    # Allocate Host Source Buffer (Random Normal)
    num_cols = 128
    total_elements = 1024 * 1024 * data_size_mb // np.dtype(np.float32).itemsize
    rows = total_elements // num_cols
    host_data = np.random.normal(size=(rows, num_cols)).astype(np.float32)

    print(
        f"Benchmarking Transfer (Smart Chunking) with Data Size: {data_size_mb} MB on"
        f" {num_devices} devices for {num_runs} iterations"
    )

    # Setup Mesh Sharding (1D)
    mesh = sharding.Mesh(
        np.array(tpu_devices).reshape((num_devices,)), axis_names=("x",)
    )
    data_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("x"))
    
    # --- ADAPTIVE WARM UP ---
    _run_warmup(host_data, data_sharding, data_size_mb)

    # Pre-calculate sharding info
    dummy_put = jax.device_put(host_data[:num_devices], data_sharding)
    target_devices = [s.device for s in dummy_put.addressable_shards]
    dummy_put.delete()

    host_shards = np.split(host_data, num_devices, axis=0)

    # Performance Lists
    h2d_perf, d2h_perf = [], []

    # --- SMART CHUNKING CONFIG ---
    def _search_runner_h2d(chunk_count):
        return _run_h2d_chunked(
            host_shards, target_devices, num_devices, chunk_count
        )

    chunks_per_device_h2d = _find_optimal_chunk_size(
        _search_runner_h2d, num_devices, data_size_mb
    )

    def _search_runner_d2h(chunk_count):
        return _run_d2h_chunked(
            host_data, data_sharding, num_devices, chunk_count
        )

    chunks_per_device_d2h = _find_optimal_chunk_size(
        _search_runner_d2h, num_devices, data_size_mb
    )
        
    # Profiling Context
    if trace_dir:
        # Create unique subdirectory for smart chunking traces
        trace_dir_smart = os.path.join(trace_dir, "smart_chunking")
        profiler_context = jax.profiler.trace(trace_dir_smart)
    else:
        profiler_context = contextlib.nullcontext()

    with profiler_context:
        for i in range(num_runs):
            if trace_dir:
                step_context = jax.profiler.StepTraceAnnotation("smart_chunking", step_num=i)
            else:
                step_context = contextlib.nullcontext()
            
            with step_context:
                h2d_ms, d2h_ms = _run_chunked(
                    host_data, data_sharding, host_shards, target_devices, 
                    num_devices, chunks_per_device_h2d, chunks_per_device_d2h
                )
                h2d_perf.append(h2d_ms)
                d2h_perf.append(d2h_ms)

    del host_data, host_shards
    gc.collect()

    return {
        "H2D_Bandwidth": h2d_perf,
        "D2H_Bandwidth": d2h_perf,
        "Chunk_Count_H2D": chunks_per_device_h2d,
        "Chunk_Count_D2H": chunks_per_device_d2h,
        "Thread_Count_H2D": num_devices * chunks_per_device_h2d,
        "Thread_Count_D2H": num_devices * chunks_per_device_d2h,
    }


def benchmark_host_device(
    num_devices: int,
    data_size_mb: int,
    num_runs: int = 100,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks H2D/D2H transfer using simple device_put/device_get (Simple Baseline)."""
    tpu_devices = get_tpu_devices(num_devices)
    
    total_elements = 1024 * 1024 * data_size_mb // np.dtype(np.float32).itemsize
    
    # Allocate Host Source Buffer (Random Normal for Simple)
    num_cols = 128
    rows = total_elements // num_cols
    host_data = np.random.normal(size=(rows, num_cols)).astype(np.float32)
    
    print(
        f"Benchmarking (Simple Baseline) Transfer with Data Size: {data_size_mb} MB on"
        f" {num_devices} devices for {num_runs} iterations"
    )

    # Setup Mesh Sharding (1D)
    mesh = sharding.Mesh(
        np.array(tpu_devices).reshape((num_devices,)), axis_names=("x",)
    )
    # Shard the 1D array across "x"
    partition_spec = sharding.PartitionSpec("x")
    
    data_sharding = sharding.NamedSharding(mesh, partition_spec)
    
    h2d_perf, d2h_perf = [], []

    # Profiling Context
    if trace_dir:
        # Create unique subdirectory for simple baseline traces
        trace_dir_simple = os.path.join(trace_dir, "simple_baseline")
        profiler_context = jax.profiler.trace(trace_dir_simple)
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
                _ = jax.device_get(device_array)
                
                t3 = time.perf_counter()
                d2h_perf.append((t3 - t2) * 1000)
                
                device_array.delete()

    return {
        "H2D_Bandwidth_ms": h2d_perf,
        "D2H_Bandwidth_ms": d2h_perf,
    }

def benchmark_host_device_calculate_metrics(
    num_devices: int = None,
    data_size_mb: int = 0,
    H2D_Bandwidth_ms: List[float] = None,
    D2H_Bandwidth_ms: List[float] = None,
    H2D_Bandwidth: List[float] = None,
    D2H_Bandwidth: List[float] = None,
    Chunk_Count: int = None,
    Thread_Count: int = None,
    Chunk_Count_H2D: int = None,
    Chunk_Count_D2H: int = None,
    Thread_Count_H2D: int = None,
    Thread_Count_D2H: int = None,
    mesh_shape: str = None,
    **kwargs
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculates metrics for Host-Device transfer (supports both Simple and Smart)."""
    
    # Gather potential metadata
    metadata = {}
    if num_devices is not None:
        metadata["num_devices"] = num_devices
    if mesh_shape is not None:
        metadata["mesh_shape"] = mesh_shape
    metadata["data_size_mb"] = data_size_mb # Uniform key
    if Chunk_Count is not None:
        metadata["Chunk_Count"] = Chunk_Count
    if Thread_Count is not None:
        metadata["Thread_Count"] = Thread_Count
    if Chunk_Count_H2D is not None:
        metadata["Chunk_Count_H2D"] = Chunk_Count_H2D
    if Chunk_Count_D2H is not None:
        metadata["Chunk_Count_D2H"] = Chunk_Count_D2H
    if Thread_Count_H2D is not None:
        metadata["Thread_Count_H2D"] = Thread_Count_H2D
    if Thread_Count_D2H is not None:
        metadata["Thread_Count_D2H"] = Thread_Count_D2H
    
    # Normalize inputs
    # Simple uses _ms suffix, Smart uses no suffix
    h2d_list = H2D_Bandwidth_ms if H2D_Bandwidth_ms is not None else H2D_Bandwidth
    d2h_list = D2H_Bandwidth_ms if D2H_Bandwidth_ms is not None else D2H_Bandwidth
    
    metrics = {}
    
    if h2d_list:
        bw_list = [
            ((data_size_mb / 1024) / (ms / 1000)) if ms > 0 else 0.0 
            for ms in h2d_list
        ]
        stats_bw = MetricsStatistics(bw_list, "H2D_bw (GiB/s)")
        metrics.update(stats_bw.serialize_statistics())

    if d2h_list:
        bw_list = [
            ((data_size_mb / 1024) / (ms / 1000)) if ms > 0 else 0.0 
            for ms in d2h_list
        ]
        stats_bw = MetricsStatistics(bw_list, "D2H_bw (GiB/s)")
        metrics.update(stats_bw.serialize_statistics())

    return metadata, metrics
