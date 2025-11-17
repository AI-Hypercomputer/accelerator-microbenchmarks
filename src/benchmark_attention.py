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
from typing import Any, Callable, Dict, Tuple
import dataclasses

from benchmark_utils import simple_timeit, MetricsStatistics
from flax import linen
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu import flash_attention as pallas_flash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
import tune_jax
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


def generate_qkv_separate_dims(
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    q_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    seed: int = 0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Generates QKV with potentially different shapes for Q, K, and V."""
    key = jax.random.PRNGKey(seed)
    key_q, key_k, key_v = jax.random.split(key, 3)
    q = jax.random.normal(key_q, (batch_size, q_heads, q_seq_len, qk_head_dim))
    k = jax.random.normal(key_k, (batch_size, kv_heads, kv_seq_len, qk_head_dim))
    v = jax.random.normal(key_v, (batch_size, kv_heads, kv_seq_len, v_head_dim))
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


def _pallas_call_hlo_pattern(mode: str, mqa: bool) -> str:
    """Generates an HLO pattern regex for filtering Pallas calls."""
    if mode not in ["fwd", "bwd", "combined"]:
        raise ValueError(f"Invalid mode: {mode}, select either 'fwd' or 'bwd'.")
    mha_or_mqa = "mqa" if mqa else "mha"
    suffix = {"fwd": "fwd", "bwd": "dkv", "combined": ""}.get(mode, "")
    return f"splash_{mha_or_mqa}_{suffix}"


def _get_tokamax_benchmark_fn(
    mask: mask_lib.Mask, config: splash.SplashConfig, mode: str, mqa: bool
) -> Callable:
    """Gets the benchmark function for Tokamax Splash Attention."""
    config = dataclasses.replace(config, use_base2_exp=True)
    if mqa:
        kernel = splash.make_splash_mqa_single_device(mask, config=config)

        @jax.jit
        def f(q, k, v):
            q = q.reshape(q.shape[:-3] + (k.shape[-3], -1) + q.shape[-2:])
            kernel_ = jax.vmap(kernel, in_axes=(0, 0, 0))  # batch vmap
            kernel_ = jax.vmap(kernel_, in_axes=(0, 0, 0))  # mqa vmap
            return kernel_(q, k, v)
    else:
        kernel = splash.make_splash_mha_single_device(mask, config=config)
        f = jax.jit(jax.vmap(kernel, in_axes=(0, 0, 0)))

    if mode == "fwd":
        return f
    if mode == "bwd":
        return jax.grad(lambda q, k, v: f(q, k, v).mean(), argnums=(0, 1, 2))
    raise ValueError(f"Invalid mode: {mode}")


def tokamax_splash_attention_benchmark(
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    q_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    mode: str = "fwd",  # One of ('fwd', 'bwd', 'combined')
    causal: bool = True,
    num_samples: int = 256,
    tune_pallas_only: bool = True,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the Tokamax Splash attention kernel."""

    if tune_pallas_only:
        event_filter_regex = _pallas_call_hlo_pattern(mode, q_heads != kv_heads)
    else:
        event_filter_regex = None

    hyperparams_override = {}
    if mode == "bwd":
        # Don't tune fwd only hyperparams
        hyperparams_override = dict(
            block_q=min(512, q_seq_len),
            block_kv=min(1024, kv_seq_len),
            block_kv_compute=min(512, kv_seq_len),
        )
    elif mode == "combined":
        mode = "bwd"

    # Generate QKV.
    q, k, v = generate_qkv_separate_dims(
        batch_size,
        q_seq_len,
        kv_seq_len,
        q_heads,
        kv_heads,
        qk_head_dim,
        v_head_dim,
    )

    # Attention mask
    mask = mask_lib.FullMask(_shape=(q_seq_len, kv_seq_len))
    if causal:
        # Pick offset for causal masks for a "representative" slice of the causal
        offset = (
            0 if q.shape[-2] == v.shape[-2] else (v.shape[-2] // 2 - q.shape[-2] // 2)
        )
        mask = mask_lib.CausalMask(shape=(q_seq_len, kv_seq_len), offset=offset)

    def attention_fn(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        block_q: int,
        block_kv: int,
        block_kv_compute: int,
        block_q_dkv: int | None,
        block_kv_dkv: int | None,
        block_kv_dkv_compute: int | None,
        block_q_dq: int | None,
        block_kv_dq: int | None,
        q_layout: splash.QKVLayout,
        k_layout: splash.QKVLayout,
        v_layout: splash.QKVLayout,
        mask: mask_lib.Mask,
        mode: str,
        mqa: bool,
    ):
        config = splash.SplashConfig(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_dkv=block_q_dkv,
            block_kv_dkv=block_kv_dkv,
            block_kv_dkv_compute=block_kv_dkv_compute,
            block_q_dq=block_q_dq,
            block_kv_dq=block_kv_dq,
            q_layout=q_layout,
            k_layout=k_layout,
            v_layout=v_layout,
        )

        f = _get_tokamax_benchmark_fn(mask, config, mode, mqa=mqa)
        return f(q, k, v)

    attention_fn = partial(
        attention_fn,
        mask=mask,
        mode=mode,
        mqa=q_heads != kv_heads,  # Determine if it's Multi-Query Attention
    )

    # Define the search space for tokamax splash attention hyperparameters.
    tiles = [256, 512, 1024, 2048]
    layouts = [splash.QKVLayout.HEAD_DIM_MINOR, splash.QKVLayout.SEQ_MINOR]
    hyperparams = {
        "block_q": tiles,
        "block_kv": tiles,
        "block_kv_compute": tiles,
        "block_q_dkv": [None],
        "block_kv_dkv": [None],
        "block_kv_dkv_compute": [None],
        "block_q_dq": [None],
        "block_kv_dq": [None],
        "q_layout": layouts,
        "k_layout": layouts,
        "v_layout": layouts,
    }

    if mode == "bwd":
        # If mode is backward, enable tuning for dKV-related block sizes.
        # These parameters are only used during the backward pass.
        hyperparams["block_q_dkv"] = tiles
        hyperparams["block_kv_dkv"] = tiles
        hyperparams["block_kv_dkv_compute"] = tiles
        hyperparams["block_q_dq"] = tiles
        hyperparams["block_kv_dq"] = tiles

    # Incorporate any potentially previously tuned hyperparameters
    hyperparams = dict(hyperparams, **hyperparams_override)

    # Prepare the attention function for tuning.
    tune_jax.CONFIG.allow_fallback_timing = False
    splash_fn = jax.jit(
        attention_fn,
        static_argnames=(
            "block_q",
            "block_kv",
            "block_kv_compute",
            "block_q_dkv",
            "block_kv_dkv",
            "block_kv_dkv_compute",
            "block_q_dq",
            "block_kv_dq",
            "q_layout",
            "k_layout",
            "v_layout",
        ),
    )

    # Tune the hyperparameters with tune_jax
    tuned_splash = tune_jax.tune(
        splash_fn,
        hyperparams=hyperparams,
        event_filter_regex=event_filter_regex,
        sample_num=num_samples,
    )

    # Run once
    output = tuned_splash(q, k, v)
    jax.block_until_ready(output)

    # Run benchmark
    time_ms_list = simple_timeit(
        tuned_splash,
        q,
        k,
        v,
        tries=num_runs,
        task="flax_attention",
        trace_dir=trace_dir,
    )
    return {"time_ms_list": time_ms_list, "output": output}


def tokamax_splash_attention_benchmark_calculate_metrics(
    # pylint: disable=unused-argument
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    q_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    mode: str,
    causal: bool,
    num_samples: int,
    tune_pallas_only: bool,
    time_ms_list: list[float],
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the tokamax splash attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)
