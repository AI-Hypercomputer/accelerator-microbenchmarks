"""Standalone single-host TPU initialization inside a multi-host slice.

Opt-in (`single_host_tpu_init` config key / CLI flag) for node-local
diagnostics. `configure_single_host_tpu()` forces a 1-host topology, skips
SliceBuilder/tpunetd and disables the cross-host XLA launch barrier. It is
process-wide and must run before the first JAX device call. The slice's ICI
routing is left untouched, so device-to-device transfers hang in this mode;
only benchmarks listed in `SUPPORTED_BENCHMARKS` are known to avoid them.
"""

import os

# Benchmarks known to perform no device-to-device transfers in this mode
# (gemm_throttling shares GeneralizedGemmBenchmark.generate_inputs). Lists
# primary names and registered aliases; keep in sync with
# `@benchmark_registry.register(..., aliases=...)`.
SUPPORTED_BENCHMARKS = frozenset({
    "gemm",
    "gemm_generalized",
    "gemm_throttling",
    "hbm",
    "hbm_bandwidth",
})

_SINGLE_HOST_ENV = {
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_SKIP_MDS_QUERY": "true",
    "TPU_WORKER_ID": "0",
    "TPU_WORKER_HOSTNAMES": "localhost",
}

_SINGLE_HOST_LIBTPU_FLAGS = {
    "--deepsea_hal_test_skip_slicebuilder": "true",
    "--deepsea_hal_test_allow_multichip_skip_slicebuilder": "true",
    "--xla_tpu_use_enhanced_launch_barrier": "false",
}


def _set_env(name: str, value: str) -> None:
  """Sets `os.environ[name]`, logging the previous value when it changes."""
  previous = os.environ.get(name)
  if previous == value:
    return
  os.environ[name] = value
  if previous is None:
    print(f"single_host_tpu_init: set {name}={value}")
  else:
    print(
        f"single_host_tpu_init: override {name}={value} (was {previous!r})"
    )


def configure_single_host_tpu() -> None:
  """Configures libtpu to initialize this host standalone. Call before JAX init."""
  for name, value in _SINGLE_HOST_ENV.items():
    _set_env(name, value)

  # Merge into LIBTPU_INIT_ARGS by flag name so pre-existing values (e.g. from
  # op_flags.yaml) are overridden rather than duplicated or left in place.
  flags: dict[str, str | None] = {}
  for token in os.environ.get("LIBTPU_INIT_ARGS", "").split():
    name, sep, value = token.partition("=")
    flags[name] = value if sep else None
  flags.update(_SINGLE_HOST_LIBTPU_FLAGS)
  _set_env(
      "LIBTPU_INIT_ARGS",
      " ".join(k if v is None else f"{k}={v}" for k, v in flags.items()),
  )
