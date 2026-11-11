"""Standalone single-host TPU initialization inside a multi-host topology.

Opt-in (`standalone_host_init` config key / CLI flag) for node-local
diagnostics. `configure_standalone_host()` forces a 1-host topology, skips
SliceBuilder/tpunetd, and disables the cross-host XLA launch barrier and
SparseCore copy offloading. It is a stateless, process-wide startup helper
called by `core/runner.py` and must run before the first JAX device call. The
topology's ICI routing is left untouched (neither intra- nor inter-host ICI
links are configured), so this mode provides no inter-device (ICI)
communication; `runner.py` rejects benchmarks whose `requires_ici` attribute is
set.
"""

import os

_STANDALONE_HOST_ENV = {
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_SKIP_MDS_QUERY": "true",
    "TPU_WORKER_ID": "0",
    "TPU_WORKER_HOSTNAMES": "localhost",
}

# JAX's `GkeTpuCluster` consults these before `TPU_WORKER_HOSTNAMES`, so a
# downstream `jax.distributed.initialize()` (e.g. in
# `platform.get_platform_info()`) must never see GKE multi-host addresses.
_STANDALONE_HOST_UNSET_ENV = (
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_ADDRESSES_PATH",
)

_STANDALONE_HOST_LIBTPU_FLAGS = {
    "--deepsea_hal_test_skip_slicebuilder": "true",
    "--deepsea_hal_test_allow_multichip_skip_slicebuilder": "true",
    # Keep barriers and copies on TensorCore; they can hang in standalone mode.
    "--xla_tpu_use_enhanced_launch_barrier": "false",
    "--xla_tpu_enable_offloading_copy_to_sparsecore": "false",
}


def _set_env(name: str, value: str) -> None:
  """Sets `os.environ[name]`, logging the previous value when it changes."""
  previous = os.environ.get(name)
  if previous == value:
    return
  os.environ[name] = value
  if previous is None:
    print(f"standalone_host_init: set {name}={value}")
  else:
    print(f"standalone_host_init: override {name}={value} (was {previous!r})")


def _unset_env(name: str) -> None:
  """Removes `os.environ[name]`, logging the previous value when present."""
  previous = os.environ.pop(name, None)
  if previous is not None:
    print(f"standalone_host_init: unset {name} (was {previous!r})")


def configure_standalone_host() -> None:
  """Configures libtpu to initialize this host standalone.

  Call before JAX init.
  """
  for name, value in _STANDALONE_HOST_ENV.items():
    _set_env(name, value)
  for name in _STANDALONE_HOST_UNSET_ENV:
    _unset_env(name)

  # Merge into LIBTPU_INIT_ARGS by flag name so pre-existing values (e.g. from
  # op_flags.yaml) are overridden rather than duplicated or left in place.
  flags: dict[str, str | None] = {}
  for token in os.environ.get("LIBTPU_INIT_ARGS", "").split():
    name, sep, value = token.partition("=")
    flags[name] = value if sep else None
  flags.update(_STANDALONE_HOST_LIBTPU_FLAGS)
  _set_env(
      "LIBTPU_INIT_ARGS",
      " ".join(k if v is None else f"{k}={v}" for k, v in flags.items()),
  )
