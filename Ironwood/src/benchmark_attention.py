"""A script to benchmark tokamax splash attention implementation.

"""

import os

# pylint: disable=g-importing-member,g-bad-import-order
from functools import partial
from typing import Any, Callable, Dict, Tuple
import dataclasses

from benchmark_utils import timeit_from_trace, MetricsStatistics
import jax
import logging
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from tokamax._src.ops.experimental.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)
import tune_jax
tune_jax.tune_logger.setLevel(logging.ERROR)

# pylint: disable=g-importing-member,g-bad-import-order

os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_dvfs_p_state=7"
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
    num_runs: int = 10,
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
        use_experimental_scheduler: bool,
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

    # Define the search space for tokamax splash attention hyperparameters.
    tiles = [256, 512, 1024, 2048, 4096, 8192]
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
        "use_experimental_scheduler": [True, False],
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
            "use_experimental_scheduler",
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
    time_ms_list = timeit_from_trace(
        tuned_splash,
        q,
        k,
        v,
        tries=num_runs,
        task="tokamax_splash_attentionatt",
        trace_dir=trace_dir,
        event_name_str_list=[
            "splash_mqa_fwd_no_residuals.1",
            "splash_mqa_dkv_no_residuals.1",
        ]
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
