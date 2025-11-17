MARKER = "!!MARKER!!"

# Predefined LIBTPU_INIT_ARGS based on TPU version and strategy
LIBTPU_ARGS_MAP = {
    "v7x": {
        "default": {
            "xla_tpu_enable_async_collective_fusion": "true",
            "xla_tpu_enable_async_collective_fusion_fuse_all_gather": "true",
            "xla_tpu_enable_async_collective_fusion_multiple_steps": "true",
            "xla_tpu_overlap_compute_collective_tc": "true",
            "xla_enable_async_all_gather": "true",
            "xla_enable_async_collective_permute": "true",
            "xla_tpu_enable_all_experimental_scheduler_features": "true",
            "xla_tpu_accumulate_into_mrb": "true",
            "xla_tpu_scoped_vmem_limit_kib": "65536",
            "xla_tpu_dvfs_p_state": "7",
            "xla_tpu_vmem_scavenging_mode": "NONE",
        },
        # Add other v6 strategies like 'latency' here if needed
    },
    "v5": {
        "default": {
            # Add v5 defaults here
        },
    },
}

def get_libtpu_args_str(tpu_version: str, strategy: str) -> str | None:
    """Retrieves LIBTPU_INIT_ARGS string for a given version and strategy."""
    try:
        args_dict = LIBTPU_ARGS_MAP[tpu_version][strategy]
        return " ".join([f"--{k}={v}" for k, v in args_dict.items()])
    except KeyError:
        print(
            f"ERROR: TPU args for version='{tpu_version}' and strategy='{strategy}'"
            " not found in LIBTPU_ARGS_MAP."
        )
        return None