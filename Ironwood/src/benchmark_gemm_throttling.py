"""Benchmarks gemm throttling."""

import os
from typing import Any, Dict

# pylint: disable=g-importing-member
from benchmark_utils import create_mesh
from benchmark_utils import get_lhs_named_shading
from benchmark_utils import get_out_sharding
from benchmark_utils import get_rhs_named_shading
from benchmark_utils import handle_based_on_sharding
from benchmark_utils import multiple_iteration_timeit_from_trace_throttling
from benchmark_utils import ShardingStrategy
from benchmark_utils import unified_flops_metrics
from common import MARKER
import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp


# pylint: disable=g-importing-member

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
    "--xla_tpu_vmem_scavenging_mode=NONE "
    "--xla_tpu_dvfs_p_state=7"
)

SHARDING_STRATEGY = ShardingStrategy.NO_SHARDING
SEED = 0
PEAK_FLOPS_PER_DEVICE = (
    2307  # TFLOP/s for single core(device) of FP8 under p_state=7
)


def gemm_throttling(
    m: int,
    k: int,
    n: int,
    num_runs: int = 1,
    dtype: jnp.dtype = jax.numpy.float8_e4m3fn,
    gap_strategy: str = "data_gen_every_iter_block_every_iter",
    trace_dir: str = None,
) -> Dict[str, Any]:
  """Benchmarks the OUT<M, N>:BF16 = IN0<M, K>:FP8 x IN1<N, K>:FP8.

  Accumulation is FP32.
  """

  def f(x, y):
    with jax.named_scope(MARKER):
      acc = jax.numpy.einsum(
          "ij,jk->ik", x, y, preferred_element_type=jnp.float32
      )
      return acc.astype(jnp.bfloat16)

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

  lhs_dtype = dtype
  rhs_dtype = dtype

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

  print("Running gemm_throttling benchmark", num_runs)
  time_ms_list = multiple_iteration_timeit_from_trace_throttling(
      jit_sharded_f,
      data_generator,
      matrix_dim=f"{m}x{n}x{k}",
      tries=num_runs,
      task="gemm_throttling",
      trace_dir=trace_dir,
      gap_strategy=gap_strategy,
  )
  return {
      "time_ms_list": time_ms_list,
  }


def gemm_throttling_calculate_metrics(
    m: int,
    k: int,
    n: int,
    gap_strategy: str,
    dtype: jnp.dtype,
    time_ms_list: list[float],
) -> Dict[str, Any]:
  # Calculate FLOPs
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
