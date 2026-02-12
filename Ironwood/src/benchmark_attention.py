"""A script to benchmark tokamax splash attention implementation.

"""

import os

# pylint: disable=g-importing-member,g-bad-import-order
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, Tuple
import dataclasses

from benchmark_utils import timeit_from_trace, MetricsStatistics
import jax
import jax.numpy as jnp

from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

# pylint: disable=g-importing-member,g-bad-import-order

os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65536"
)

SplashAttentionLookupKey = tuple[
    int, # batch_size
    int, # q_seq_len
    int, # kv_seq_len
    int, # q_heads
    int, # kv_heads
    int, # qk_head_dim
    int, # v_head_dim
    bool, # causal
]

SplashAttentionLookupValue = tuple[
    int, # block_q
    int, # block_kv
    int, # block_kv_compute
    int, # block_q_dkv
    int, # block_kv_dkv
    int, # block_kv_dkv_compute
    splash.QKVLayout, # q_layout
    splash.QKVLayout, # k_layout
    splash.QKVLayout, # v_layout
    bool, # use_experimental_scheduler
]

# Merge the tuned block size of optimal fwd and bwd
# The optimal layout and use_experimental_scheduler may be different between fwd and bwd
# Use the layout and use_experimental_scheduler optimized for fwd
SPLASH_ATTENTION_HYPERPARAMS_LOOKUP_TABLE: Dict[
    SplashAttentionLookupKey, SplashAttentionLookupValue
] = {
    (1, 4096, 4096, 128, 128, 256, 256, True): (
        2048,
        2048,
        256,
        2048,
        2048,
        512,
        splash.QKVLayout.HEAD_DIM_MINOR,
        splash.QKVLayout.SEQ_MINOR,
        splash.QKVLayout.HEAD_DIM_MINOR,
        True,
    ),
    (1, 4096, 4096, 128, 128, 256, 256, False): (
        4096,
        4096,
        512,
        4096,
        2048,
        512,
        splash.QKVLayout.HEAD_DIM_MINOR,
        splash.QKVLayout.SEQ_MINOR,
        splash.QKVLayout.HEAD_DIM_MINOR,
        True,
    ),
}

DEFAULT_SPLASH_ATTENTION_HYPERPARAMS: SplashAttentionLookupValue = (
    2048,
    2048,
    256,
    2048,
    2048,
    256,
    splash.QKVLayout.HEAD_DIM_MINOR,
    splash.QKVLayout.SEQ_MINOR,
    splash.QKVLayout.HEAD_DIM_MINOR,
    True,
)


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
    q = jax.random.normal(key_q, (batch_size, q_heads, q_seq_len, qk_head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(key_k, (batch_size, kv_heads, kv_seq_len, qk_head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(key_v, (batch_size, kv_heads, kv_seq_len, v_head_dim), dtype=jnp.bfloat16)
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
    mode: Literal["fwd", "bwd"] = "fwd",
    causal: bool = True,
    num_runs: int = 10,
    trace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmarks the Tokamax Splash attention kernel."""
    event_filter_regex = _pallas_call_hlo_pattern(mode, q_heads != kv_heads)
    # Generate QKV in shape [batch, head_num, seq_len, head_dim].
    q, k, v = generate_qkv_separate_dims(
        batch_size,
        q_seq_len,
        kv_seq_len,
        q_heads,
        kv_heads,
        qk_head_dim,
        v_head_dim,
    )

    key = (
        batch_size,
        q_seq_len,
        kv_seq_len,
        q_heads,
        kv_heads,
        qk_head_dim,
        v_head_dim,
        causal,
    )
    hyperparams: Optional[SplashAttentionLookupValue] = (
        SPLASH_ATTENTION_HYPERPARAMS_LOOKUP_TABLE.get(key, None)
    )
    has_optimized = True
    if hyperparams is None:
        print(f"{key=} is not tuned")
        has_optimized = False
        hyperparams = DEFAULT_SPLASH_ATTENTION_HYPERPARAMS
    
    (
        block_q,
        block_kv,
        block_kv_compute,
        block_q_dkv,
        block_kv_dkv,
        block_kv_dkv_compute,
        q_layout,
        k_layout,
        v_layout,
        use_experimental_scheduler,
    ) = hyperparams

    # Pad q, kv to prevent the block size are not valid
    if not has_optimized:
        def _ceiling_div(a: int, b: int) -> int:
            return (a + b - 1) // b


        def _align_to(x: int, a: int) -> int:
            return _ceiling_div(x, a) * a

        def _pad_token(t: jax.Array, size) -> jax.Array:
            # tensor is [batch_size, num_head, token, head_dim]
            result = jnp.pad(t, ((0, 0), (0, 0), (0, size), (0, 0)), constant_values=0)
            return result

        q_len = q.shape[-2]
        k_len = k.shape[-2]

        # Pad q, k, v sequence, align to block sizes
        q = _pad_token(q, _align_to(q_len, block_q) - q_len)
        k = _pad_token(k, _align_to(k_len, block_kv) - k_len)
        v = _pad_token(v, _align_to(k_len, block_kv) - k_len)

    padded_q_len = q.shape[-2]
    padded_kv_len = k.shape[-2]
    # Attention mask
    mask = mask_lib.FullMask(_shape=(padded_q_len, padded_kv_len))
    if causal:
        # Pick offset for causal masks for a "representative" slice of the causal
        offset = padded_kv_len - padded_q_len
        mask = mask_lib.CausalMask(shape=(padded_q_len, padded_kv_len), offset=offset)

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
        q_layout: splash.QKVLayout,
        k_layout: splash.QKVLayout,
        v_layout: splash.QKVLayout,
        mask: mask_lib.Mask,
        mode: str,
        mqa: bool,
        use_experimental_scheduler: bool,
    ):
        # dq kernel is not used
        config = splash.SplashConfig(
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=block_kv_compute,
            block_q_dkv=block_q_dkv,
            block_kv_dkv=block_kv_dkv,
            block_kv_dkv_compute=block_kv_dkv_compute,
            block_q_dq=None,
            block_kv_dq=None,
            q_layout=q_layout,
            k_layout=k_layout,
            v_layout=v_layout,
            use_experimental_scheduler=use_experimental_scheduler,
        )

        f = _get_tokamax_benchmark_fn(mask, config, mode, mqa=mqa)
        return f(q, k, v)

    attention_fn = partial(
        attention_fn,
        mask=mask,
        mode=mode,
        mqa=q_heads != kv_heads,  # Determine if it's Multi-Query Attention
    )

    splash_fn = jax.jit(
        attention_fn,
        static_argnames=(
            "block_q",
            "block_kv",
            "block_kv_compute",
            "block_q_dkv",
            "block_kv_dkv",
            "block_kv_dkv_compute",
            "q_layout",
            "k_layout",
            "v_layout",
            "use_experimental_scheduler",
        ),
    )

    tuned_splash = partial(
        splash_fn,
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        q_layout=q_layout,
        k_layout=k_layout,
        v_layout=v_layout,
        use_experimental_scheduler=use_experimental_scheduler,
    )

    # Run once
    output = tuned_splash(q, k, v)
    jax.block_until_ready(output)

    print("-" * 50)
    print(
        f"batch_size={batch_size}, q_seq_len={q_seq_len}, kv_seq_len={kv_seq_len}, "
        f"q_heads={q_heads}, kv_heads={kv_heads}, qk_head_dim={qk_head_dim}, "
        f"v_head_dim={v_head_dim}, mode={mode}, causal={causal}"
    )
    print(f"{hyperparams=}")
    print("-" * 50)

    # Run benchmark
    time_ms_list = timeit_from_trace(
        tuned_splash,
        q,
        k,
        v,
        tries=num_runs,
        task="tokamax_splash_attentionatt",
        trace_dir=trace_dir,
        event_name_str_list=[
            f"{event_filter_regex}_no_residuals.1",
        ]
    )
    return {
        "time_ms_list": time_ms_list,
        "output": output,
        "has_optimized": has_optimized,
    }


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
    time_ms_list: list[float],
    has_optimized: bool,
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
    """Gathers metrics for the tokamax splash attention benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    return get_metrics_helper(params)
