"""Benchmarks Host-to-Device and Device-to-Host transfer performance (Simple Baseline)."""

import time
import os
from typing import Any, Dict, Tuple, List

import jax
from jax import numpy as jnp
import numpy as np
from benchmark_utils import MetricsStatistics
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils


libtpu_init_args = [
    "--xla_tpu_dvfs_p_state=7",
]
os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
# 64 GiB
os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "68719476736"
os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "68719476736"


import abc

class TransferStrategy(abc.ABC):
    """Abstract base class for transfer strategies."""

    def __init__(self, trace_dir: str = None):
        self.trace_dir = trace_dir
        self.h2d_perf = []
        self.d2h_perf = []

    @abc.abstractmethod
    def setup(self, data_size_mib: int, host_data: np.ndarray, devices: List[jax.Device]):
        """Perform one-time setup before the benchmark loop."""
        pass

    @abc.abstractmethod
    def run_h2d(self, host_data: np.ndarray, i: int) -> Any:
        """Run H2D transfer for iteration i."""
        pass

    @abc.abstractmethod
    def run_d2h(self, device_data: Any, i: int):
        """Run D2H transfer for iteration i."""
        pass

    @abc.abstractmethod
    def teardown(self):
        """Clean up resources after the benchmark loop."""
        pass


class SimpleTransfer(TransferStrategy):
    """Simple device_put/device_get strategy."""

    def setup(self, data_size_mib: int, host_data: np.ndarray):
        pass

    def run_h2d(self, host_data: np.ndarray, i: int) -> Any:
        t0 = time.perf_counter()
        device_array = jax.device_put(host_data)
        device_array.block_until_ready()
        t1 = time.perf_counter()
        
        # Verify H2D shape
        assert device_array.shape == host_data.shape
        self.h2d_perf.append((t1 - t0) * 1000)
        return device_array

    def run_d2h(self, device_data: Any, i: int):
        t2 = time.perf_counter()
        _ = jax.device_get(device_data)
        t3 = time.perf_counter()
        self.d2h_perf.append((t3 - t2) * 1000)
        device_data.delete()

    def teardown(self):
        pass


class PipelinedTransfer(TransferStrategy):
    """Pipelined transfer using chunking."""

    def setup(self, data_size_mib: int, host_data: np.ndarray):
        self.target_chunk_size_mib = 16
        num_devices_to_perform_h2d = 2
        self.target_devices = jax.devices()[:num_devices_to_perform_h2d]
        self.num_devices = len(self.target_devices)
        
        data_per_dev = data_size_mib / self.num_devices
        chunks_per_dev = int(data_per_dev / self.target_chunk_size_mib)
        self.chunks_per_dev = max(1, chunks_per_dev)
        if self.chunks_per_dev == 1:
            print(f"Warning: Data size is not larger than target chunk size, falling back to standard JAX put.")

    def run_h2d(self, host_data: np.ndarray, i: int) -> Any:
        if self.chunks_per_dev > 1:
            chunks = np.array_split(host_data, self.chunks_per_dev * self.num_devices, axis=0)
            tensors_on_device = []
            
            t0 = time.perf_counter()
            for idx, chunk in enumerate(chunks):
                dev = self.target_devices[idx % self.num_devices]
                tensors_on_device.append(jax.device_put(chunk, dev))
            
            for device_tensor in tensors_on_device:
                device_tensor.block_until_ready()
            t1 = time.perf_counter()
            
            self.h2d_perf.append((t1 - t0) * 1000)
            del chunks
            return tensors_on_device
        else:
            t0 = time.perf_counter()
            result = jax.device_put(host_data, self.target_devices[0])
            result.block_until_ready()
            t1 = time.perf_counter()
            self.h2d_perf.append((t1 - t0) * 1000)
            return result

    def run_d2h(self, device_data: Any, i: int):
        t2 = time.perf_counter()
        if isinstance(device_data, list):
            _ = jax.device_get(device_data)
            for device_tensor in device_data:
                device_tensor.delete()
        else:
            _ = jax.device_get(device_data)
            device_data.delete()
        t3 = time.perf_counter()
        self.d2h_perf.append((t3 - t2) * 1000)

    def teardown(self):
        pass


class PinnedMemoryTransfer(TransferStrategy):
    """Pinned memory host-to-device with parallelized device-to-host transfer."""

    def setup(self, data_size_mib: int, host_data: np.ndarray):
        num_devices_to_perform_h2d = 2
        target_devices = jax.devices()[:num_devices_to_perform_h2d]
        
        mesh = Mesh(target_devices, ('x',))
        partition_spec = PartitionSpec('x')
        host_sharding = NamedSharding(mesh, partition_spec, memory_kind='pinned_host')
        self.pinned_device_sharding = NamedSharding(mesh, partition_spec)

        print("  Allocating Pinned Host Data...", flush=True)
        self.pinned_host_input = jax.device_put(host_data, host_sharding)
        self.pinned_host_input.block_until_ready()

    def run_h2d(self, host_data: np.ndarray, i: int) -> Any:
        t_transfer_start = time.perf_counter()
        device_array = jax.device_put(self.pinned_host_input, self.pinned_device_sharding)
        device_array.block_until_ready()
        t_transfer_end = time.perf_counter()
        
        self.h2d_perf.append((t_transfer_end - t_transfer_start) * 1000)
        return device_array

    def run_d2h(self, device_data: Any, i: int):
        t2 = time.perf_counter()
        # Fetch addressable shards to enable pipelined D2H
        _ = jax.device_get([s.data for s in device_data.addressable_shards])
        t3 = time.perf_counter()
        self.d2h_perf.append((t3 - t2) * 1000)
        device_data.delete()

    def teardown(self):
        if hasattr(self, 'pinned_host_input'):
             self.pinned_host_input.delete()


def benchmark_host_device(
    data_size_mib: int,
    transfer_type: str,
    num_runs: int = 100,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks H2D/D2H transfer using device_put/device_get."""
    
    num_elements = 1024 * 1024 * data_size_mib // np.dtype(np.float32).itemsize
    
    # Allocate Host Source Buffer
    column = 128
    host_data = np.random.normal(size=(num_elements // column, column)).astype(np.float32)

    print(
        f"Benchmarking Transfer with Data Size: {data_size_mib} MB for {num_runs} iterations with {transfer_type=}",
        flush=True
    )

    strategies = {
        "simple": SimpleTransfer,
        "pipelined": PipelinedTransfer,
        "pinned_memory": PinnedMemoryTransfer,
    }

    if transfer_type not in strategies:
        raise ValueError(f"Unknown transfer_type: {transfer_type}. Available: {list(strategies.keys())}")

    strategy = strategies[transfer_type](trace_dir)
    strategy.setup(data_size_mib, host_data)

    # Profiling Context
    import contextlib
    if trace_dir:
        profiler_context = jax.profiler.trace(trace_dir)
    else:
        profiler_context = contextlib.nullcontext()

    with profiler_context:
        # Warmup
        for _ in range(2):
            device_array = jax.device_put(host_data)
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
                device_data = strategy.run_h2d(host_data, i)
                strategy.run_d2h(device_data, i)

    strategy.teardown()

    return {
        "H2D_Bandwidth_ms": strategy.h2d_perf,
        "D2H_Bandwidth_ms": strategy.d2h_perf,
    }

def benchmark_host_device_calculate_metrics(
    data_size_mib: int,
    transfer_type: str,
    H2D_Bandwidth_ms: List[float],
    D2H_Bandwidth_ms: List[float],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculates metrics for Host-Device transfer."""
    params = locals().items()
    
    # Filter out list params from metadata to avoid explosion
    metadata_keys = {
        "data_size_mib", 
    }
    metadata = {k: v for k, v in params if k in metadata_keys}
    metadata["dtype"] = "float32"
    metadata["transfer_type"] = transfer_type
    
    metrics = {}
    
    def add_metric(name, ms_list):
        # Report Bandwidth (GiB/s)
        # Handle division by zero if ms is 0
        bw_list = [
            ((data_size_mib / 1024) / (ms / 1000)) if ms > 0 else 0.0 
            for ms in ms_list
        ]
        stats_bw = MetricsStatistics(bw_list, f"{name}_bw (GiB/s)")
        print(
            f"  {name}_bw (GiB/s) median: {stats_bw.statistics['p50']}, P95: {stats_bw.statistics['p95']}", 
            flush=True
        )
        metrics.update(stats_bw.serialize_statistics())

    add_metric("H2D", H2D_Bandwidth_ms)
    add_metric("D2H", D2H_Bandwidth_ms)

    return metadata, metrics
