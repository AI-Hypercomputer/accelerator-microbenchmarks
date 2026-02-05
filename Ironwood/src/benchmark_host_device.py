    """Benchmarks Host-to-Device and Device-to-Host transfer performance (Simple Baseline)."""

    import time
    import os
    from typing import Any, Dict, Tuple, List

    import jax
    from jax import numpy as jnp
    import numpy as np
    from benchmark_utils import MetricsStatistics


    libtpu_init_args = [
        "--xla_tpu_dvfs_p_state=7",
    ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)
    # 64 GiB
    os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "68719476736"
    os.environ["TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES"] = "68719476736"


    def benchmark_host_device(
        h2d_type: str,
        data_size_mib: int,
        num_runs: int = 100,
        trace_dir: str = None,
    ) -> Dict[str, Any]:
        """Benchmarks H2D/D2H transfer using device_put/device_get."""
        
        num_elements = 1024 * 1024 * data_size_mib // np.dtype(np.float32).itemsize
        
        # Allocate Host Source Buffer
        column = 128
        host_data = np.random.normal(size=(num_elements // column, column)).astype(np.float32)
        
        # Used in pipelined flow
        # TODO: turn into a param
        num_devices_to_perform_h2d = 1
        target_devices = jax.devices()[:num_devices_to_perform_h2d]

<<<<<<< Updated upstream
    print(
        f"Benchmarking Transfer with Data Size: {data_size_mib} MB for {num_runs} iterations with {h2d_type=}",
        flush=True
    )

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
                 # H2D
                if h2d_type == "simple":
                    t0 = time.perf_counter()
                    # Simple device_put
                    device_array = jax.device_put(host_data)
                    device_array.block_until_ready()
                    t1 = time.perf_counter()
                    
                    # Verify H2D shape
                    assert device_array.shape == host_data.shape

                    h2d_perf.append((t1 - t0) * 1000)
                
                    # D2H
                    t2 = time.perf_counter()
                    
                    # Simple device_get
                    # Note: device_get returns a numpy array (copy)
                    _ = jax.device_get(device_array)
                    
                    t3 = time.perf_counter()
                    d2h_perf.append((t3 - t2) * 1000)
                    
                    device_array.delete()
                elif h2d_type == "pipelined":
                    target_chunk_size_mib = 16  # Sweet spot from profiling
                    num_devices = len(target_devices)

                    tensors_on_device = []
                    
                    # Calculate chunks per device
                    data_per_dev = data_size_mib / num_devices
                    chunks_per_dev = int(data_per_dev / target_chunk_size_mib)
                    chunks_per_dev = max(1, chunks_per_dev)

                    chunks = np.array_split(host_data, chunks_per_dev * num_devices, axis=0)

                    t0 = time.perf_counter()
                    if chunks_per_dev > 1:    
                        # We need to map chunks to the correct device
                        # This simple example assumes chunks are perfectly divisible and ordered
                        # In production, use `jax.sharding` mesh logic for complex layouts

                        # approach 1: simple for loop
                        for idx, chunk in enumerate(chunks):
                            if num_devices > 1:
                                dev = target_devices[idx % num_devices]
                            else:
                                dev = target_devices[0]
                            tensors_on_device.append(jax.device_put(chunk, dev))
                        # Re-assemble array
                        result = jnp.vstack(tensors_on_device)
                        # Wait for all chunks to be transferred
                        result.block_until_ready()

                        # approach 2: generator (slightly less overhead)
                        # def chunk_generator(num_devices, chunks_per_dev):
                        #     for n in range(chunks_per_dev):
                        #         for d in range(num_devices):
                        #             # 1. Get the specific small chunk
                        #             chunk = chunks[d*chunks_per_dev+n]

                        #             # 2. Trigger an individual DMA transfer for this specific chunk
                        #             # This is where NUMA-local memory access matters
                        #             yield jax.device_put(chunk, target_devices[d])

                        # # Re-assemble array
                        # result = jnp.vstack(list(chunk_generator(num_devices, chunks_per_dev)))
                        # # Wait for all chunks to be transferred
                        # result.block_until_ready()
                    else:
                        print(f"Warning: {data_size_mib=} is not larger than {target_chunk_size_mib=}, falling back to standard JAX put.")
                        # Fallback to standard JAX put for small data
                        result = jax.device_put(host_data, target_devices[0])
                        result.block_until_ready()

                    t1 = time.perf_counter()
                    h2d_perf.append((t1 - t0) * 1000)

                    # D2H
                    t2 = time.perf_counter()
                    # Simple device_get
                    # Note: device_get returns a numpy array (copy)
                    _ = jax.device_get(result)

                    t3 = time.perf_counter()
                    if not np.allclose(result, host_data):
                        print("pipelined result not equal to host_data")
                    d2h_perf.append((t3 - t2) * 1000)

                    for r in tensors_on_device:
                        r.delete()
                    del tensors_on_device

    return {
        "H2D_Bandwidth_ms": h2d_perf,
        "D2H_Bandwidth_ms": d2h_perf,
    }

def benchmark_host_device_calculate_metrics(
    data_size_mib: int,
    H2D_Bandwidth_ms: List[float],
    D2H_Bandwidth_ms: List[float],
    h2d_type: str = "simple",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Calculates metrics for Host-Device transfer."""
    params = locals().items()
    
    # Filter out list params from metadata to avoid explosion
    metadata_keys = {
        "data_size_mib", 
    }
    metadata = {k: v for k, v in params if k in metadata_keys}
    metadata["dtype"] = "float32"
    metadata["h2d_type"] = h2d_type
    
    metrics = {}
    
    def add_metric(name, ms_list):
        # Report Bandwidth (GiB/s)
        # Handle division by zero if ms is 0
        bw_list = [
            ((data_size_mib / 1024) / (ms / 1000)) if ms > 0 else 0.0 
            for ms in ms_list
        ]
        stats_bw = MetricsStatistics(bw_list, f"{name}_bw (GiB/s)")
=======
>>>>>>> Stashed changes
        print(
            f"Benchmarking Transfer with Data Size: {data_size_mib} MB for {num_runs} iterations with {h2d_type=}",
            flush=True
        )

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
                    # H2D
                    if h2d_type == "simple":
                        t0 = time.perf_counter()
                        # Simple device_put
                        device_array = jax.device_put(host_data)
                        device_array.block_until_ready()
                        t1 = time.perf_counter()
                        
                        # Verify H2D shape
                        assert device_array.shape == host_data.shape

                        h2d_perf.append((t1 - t0) * 1000)
                    
                        # D2H
                        t2 = time.perf_counter()
                        
                        # Simple device_get
                        # Note: device_get returns a numpy array (copy)
                        _ = jax.device_get(device_array)
                        
                        t3 = time.perf_counter()
                        d2h_perf.append((t3 - t2) * 1000)
                        
                        device_array.delete()
                    elif h2d_type == "pipelined":
                        target_chunk_size_mib = 16  # Sweet spot from profiling
                        num_devices = len(target_devices)

                        tensors_on_device = []
                        
                        # Calculate chunks per device
                        data_per_dev = data_size_mib / num_devices
                        chunks_per_dev = int(data_per_dev / target_chunk_size_mib)
                        chunks_per_dev = max(1, chunks_per_dev)

                        chunks = np.array_split(host_data, chunks_per_dev * num_devices, axis=0)

                        t0 = time.perf_counter()
                        if chunks_per_dev > 1:    
                            # We need to map chunks to the correct device
                            # This simple example assumes chunks are perfectly divisible and ordered
                            # In production, use `jax.sharding` mesh logic for complex layouts

                            # approach 1: simple for loop
                            for idx, chunk in enumerate(chunks):
                                if num_devices > 1:
                                    dev = target_devices[idx % num_devices]
                                else:
                                    dev = target_devices[0]
                                tensors_on_device.append(jax.device_put(chunk, dev))
                            # Re-assemble array
                            # result = jnp.vstack(tensors_on_device)
                            # Wait for all chunks to be transferred
                            # result.block_until_ready()
                            
                            # Don't re-assemble
                            for tensor in tensors_on_device:
                                tensor.block_until_ready()

                            # approach 2: generator (slightly less overhead)
                            # def chunk_generator(num_devices, chunks_per_dev):
                            #     for n in range(chunks_per_dev):
                            #         for d in range(num_devices):
                            #             # 1. Get the specific small chunk
                            #             chunk = chunks[d*chunks_per_dev+n]

                            #             # 2. Trigger an individual DMA transfer for this specific chunk
                            #             # This is where NUMA-local memory access matters
                            #             yield jax.device_put(chunk, target_devices[d])

                            # # Re-assemble array
                            # result = jnp.vstack(list(chunk_generator(num_devices, chunks_per_dev)))
                            # # Wait for all chunks to be transferred
                            # result.block_until_ready()
                        else:
                            print(f"Warning: {data_size_mib=} is not larger than {target_chunk_size_mib=}, falling back to standard JAX put.")
                            # Fallback to standard JAX put for small data
                            result = jax.device_put(host_data, target_devices[0])
                            result.block_until_ready()

                        t1 = time.perf_counter()
                        h2d_perf.append((t1 - t0) * 1000)

                        # D2H
                        t2 = time.perf_counter()
                        # Simple device_get
                        # Note: device_get returns a numpy array (copy)
                        # result = jnp.vstack(tensors_on_device)
                        # _ = jax.device_get(result)
                        # del tensors_on_device

                        # device_put instead
                        tensors_on_host = []
                        for tensor in tensors_on_device:
                            tensors_on_host.append(jax.device_put(x, jax.devices("cpu")[0]))
                        for tensor in tensors_on_host:
                            tensor.block_until_ready()
                        
                        t3 = time.perf_counter()
                        if not np.allclose(result, host_data):
                            print("pipelined result not equal to host_data")
                        d2h_perf.append((t3 - t2) * 1000)

                        for r in tensors_on_device:
                            r.delete()
                        del tensors_on_device
                        for r in tensors_on_host:
                            r.delete()
                        del tensors_on_host

        return {
            "H2D_Bandwidth_ms": h2d_perf,
            "D2H_Bandwidth_ms": d2h_perf,
        }

    def benchmark_host_device_calculate_metrics(
        data_size_mib: int,
        H2D_Bandwidth_ms: List[float],
        D2H_Bandwidth_ms: List[float],
        h2d_type: str = "simple",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Calculates metrics for Host-Device transfer."""
        params = locals().items()
        
        # Filter out list params from metadata to avoid explosion
        metadata_keys = {
            "data_size_mib", 
        }
        metadata = {k: v for k, v in params if k in metadata_keys}
        metadata["dtype"] = "float32"
        
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
