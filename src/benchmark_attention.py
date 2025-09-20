"""A script to benchmark different attention implementations.

1. naive_attention: mha attention calculation with no optimization. This is the
baseline. The benchmarks only consider the core attention mechanism—this
includes the computation of attention weights and applying them to the value
vectors (v) but excludes the linear projections used to generate the query (q),
key (k), and value (v) vectors.
2. pallas_flash_attention_benchmark: attention with the pallas flash
attention kernel.
(https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py)
3. splash_attention_benchmark: attention with the splash attention kernel.
    (https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention)
4. flax_nnx_attention_benchmark: attention with the flax nnx attention library.
5. flax_linen_attention_benchmark: attention with the flax linen attention
library.
6. keras_attention_benchmark: attention with the keras attention library.
"""

# pylint: disable=g-importing-member,g-bad-import-order
from functools import partial
import os
from typing import Any, Dict, Tuple

from benchmark_utils import simple_timeit, MetricsStatistics
from flax import linen
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu import flash_attention as pallas_flash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
import numpy as np

# pylint: disable=g-importing-member,g-bad-import-order

os.environ["KERAS_BACKEND"] = "jax"
import keras  # pylint: disable=g-bad-import-order,g-import-not-at-top

# Tunable parameters for splash attention.
# Kernel block sizes.
SPLASH_ATTENTION_GLOBAL_BLOCK_QKV = 128
# Setting this parameter will enable local sliding attention.
SPLASH_ATTENTION_SLIDING_WINDOW_SIZE = None


def generate_qkv(
    batch: int, seq_len: int, d_model: int, num_heads: int, seed: int = 0
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Generates QKV with the given shape."""
    key = jax.random.PRNGKey(seed)
    key_q, key_k, key_v = jax.random.split(key, 3)
    head_dim = d_model // num_heads
    q = jax.random.normal(key_q, (batch, num_heads, seq_len, head_dim))
    k = jax.random.normal(key_k, (batch, num_heads, seq_len, head_dim))
    v = jax.random.normal(key_v, (batch, num_heads, seq_len, head_dim))
    return q, k, v


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
    metrics = {}
    time_ms_statistics = MetricsStatistics(
        metrics_list=dict(params)["time_ms_list"], metrics_name="time_ms"
    )
    metrics.update(time_ms_statistics.serialize_statistics())
    return metadata, metrics


def naive_attention_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool = True,
    scale: bool = False,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Naive attention benchmark."""

    @partial(jax.jit, static_argnames=["causal", "scale"])
    def f(q, k, v, causal, scale):
        # qkv shape: ('batch', 'heads', 'length', 'kv')
        _, _, _, k_kv_size = k.shape
        _, _, seq_lengh, _ = q.shape
        scale_factor = 1.0
        if scale:
            scale_factor = 1.0 / jnp.sqrt(k_kv_size)
        weights_unnormalized = jax.numpy.einsum("BHSD,BHTD->BHST", q, k) * scale_factor
        if causal:
            weights_unnormalized_to_zero_out = jax.numpy.triu(
                jax.numpy.ones((seq_lengh, seq_lengh), jax.numpy.bfloat16), 1
            )
            weights = jax.nn.softmax(
                weights_unnormalized - 1e6 * weights_unnormalized_to_zero_out
            )
        else:
            weights = jax.nn.softmax(weights_unnormalized)
        return jax.numpy.einsum("BHST,BHTD->BHSD", weights, v)

    # Generate QKV.
    q, k, v = generate_qkv(batch, seq_len, d_model, num_heads)
    # Run once
    output = f(q, k, v, causal, scale)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        f,
        q,
        k,
        v,
        causal,
        scale,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="naive_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def naive_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool,
    scale: bool,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the naive attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)


def pallas_flash_attention_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool = True,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the Pallas flash attention kernel."""

    @partial(jax.jit, static_argnames=["causal"])
    def f(q, k, v, causal):
        return pallas_flash_attention.mha_reference(
            q, k, v, ab=None, segment_ids=None, causal=causal
        )

    # Generate QKV.
    q, k, v = generate_qkv(batch, seq_len, d_model, num_heads)
    # Run once
    output = f(q, k, v, causal)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        f,
        q,
        k,
        v,
        causal,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="pallas_flash_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def pallas_flash_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the pallas flash attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)


def splash_attention_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool = True,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the Splash attention kernel."""

    @partial(jax.jit, static_argnames=["causal"])
    def f(q, k, v, causal):
        # ('batch', 'heads', 'length', 'kv')
        _, _, seq_len, _ = q.shape
        sliding_window_size = SPLASH_ATTENTION_SLIDING_WINDOW_SIZE
        global_block_q = global_block_kv = SPLASH_ATTENTION_GLOBAL_BLOCK_QKV
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(global_block_q, seq_len),
            block_kv=min(global_block_kv, k.shape[2]),
        )
        mask = splash_attention_mask.FullMask(_shape=(seq_len, seq_len))
        if causal:
            mask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))

        # Apply local masking if local sliding attention is enabled.
        if sliding_window_size is not None:
            mask &= splash_attention_mask.LocalMask(
                shape=(seq_len, seq_len),
                window_size=(sliding_window_size, sliding_window_size),
                offset=0,
            )

        # Create multi-head mask
        multi_head_mask = splash_attention_mask.MultiHeadMask(
            masks=(mask,) * q.shape[1]
        )
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=multi_head_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
        )
        output = jax.vmap(splash_kernel)(q, k, v)

        return output

    # Generate QKV.
    q, k, v = generate_qkv(batch, seq_len, d_model, num_heads)
    # Run once
    output = f(q, k, v, causal)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        f,
        q,
        k,
        v,
        causal,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="splash_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def splash_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the splash attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)


def flax_nnx_attention_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the Flax nnx attention."""

    @jax.jit
    def f(q, k, v):
        output = nnx.dot_product_attention(q, k, v)
        return output

    # Generate QKV.
    q, k, v = generate_qkv(batch, seq_len, d_model, num_heads)

    # Flax q,k,v shape: [batch, q_length, num_heads, qk_depth_per_head]
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Run once
    output = f(q, k, v)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        f,
        q,
        k,
        v,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="flax_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def flax_nnx_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the flax nnx attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)


def flax_linen_attention_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the Flax linen attention."""

    @jax.jit
    def f(q, k, v):
        output = linen.dot_product_attention(q, k, v)
        return output

    # Generate QKV.
    q, k, v = generate_qkv(batch, seq_len, d_model, num_heads)
    # Flax q,k,v shape: (batch, q_length, num_heads, qk_depth_per_head)
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))

    # Run once
    output = f(q, k, v)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        f,
        q,
        k,
        v,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="flax_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def flax_linen_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the flax linen attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)


def keras_attention_benchmark(
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    causal: bool = False,
    num_runs: int = 1,
    trace_dir: str = None,
    warmup_tries: int = 10,
) -> Dict[str, Any]:
    """Benchmarks the Flax linen attention."""

    # Generate QKV.
    q, k, v = generate_qkv(batch, seq_len, d_model, num_heads)

    # Transpose to: (batch, q_length, num_heads, size_per_head)
    q = np.transpose(q, (0, 2, 1, 3))
    k = np.transpose(k, (0, 2, 1, 3))
    v = np.transpose(v, (0, 2, 1, 3))
    _, _, num_heads, head_dim = q.shape

    # TODO(qinyiyan): Use the flash attention mode:
    # https://github.com/keras-team/keras/blob/b0b9d041833dce623871c1e233ea24316d4471be/keras/src/layers/attention/multi_head_attention.py#L56

    layer = keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=head_dim, value_dim=head_dim
    )

    @partial(jax.jit, static_argnames=["causal"])
    def f(q, k, v, causal):
        output = layer(query=q, key=k, value=v, use_causal_mask=causal)
        return output

    # Run once
    output = f(q, k, v, causal)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        f,
        q,
        k,
        v,
        causal,
        warmup_tries=warmup_tries,
        tries=num_runs,
        task="keras_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def keras_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the keras attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)
