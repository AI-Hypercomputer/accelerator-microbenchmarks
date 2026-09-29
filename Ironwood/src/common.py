"""Common constants and helpers for the microbenchmarks."""

import os
from typing import Iterable

MARKER = "!!MARKER!!"

# libtpu flags that turn this host into a standalone single-host slice.
# On a multi-host slice, GKE / the TPU VM metadata describe the whole slice,
# so libtpu waits for every host to join before the TPU backend comes up.
# libtpu appends LIBTPU_INIT_ARGS after the flags it derives from the
# environment, so these take precedence. With host bounds of 1,1,1 libtpu
# uses task id 0 and a localhost-only slice builder, and ignores the other
# hosts listed in TPU_WORKER_HOSTNAMES. Chips-per-host bounds are left as
# provided, since they already describe a single host.
SINGLE_HOST_LIBTPU_ARGS = (
    "--deepsea_host_bounds=1,1,1",
    "--deepsea_wrap=false,false,false",
    "--deepsea_twist=false",
)
_single_host_mode_enabled = False


def set_libtpu_init_args(libtpu_init_args: Iterable[str]):
    """Sets LIBTPU_INIT_ARGS, keeping the single-host flags when enabled.

    Benchmarks that set LIBTPU_INIT_ARGS after run_benchmark has started
    (e.g. inside a benchmark function) must use this instead of writing
    os.environ directly, so that single_host mode keeps working.

    Args:
      libtpu_init_args: The libtpu flags to set.
    """
    libtpu_init_args = list(libtpu_init_args)
    if _single_host_mode_enabled:
        libtpu_init_args += [
            arg
            for arg in SINGLE_HOST_LIBTPU_ARGS
            if arg not in libtpu_init_args
        ]
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(libtpu_init_args)


def enable_single_host_mode():
    """Makes the TPU backend come up as a standalone single-host slice.

    Lets benchmarks run on one host of a multi-host slice without the other
    hosts taking part. Must be called before the TPU backend is initialized,
    and after any benchmark module that sets LIBTPU_INIT_ARGS at import time
    has been imported.
    """
    global _single_host_mode_enabled  # pylint: disable=global-statement
    _single_host_mode_enabled = True
    set_libtpu_init_args(os.environ.get("LIBTPU_INIT_ARGS", "").split())


def single_host_mode_enabled() -> bool:
    """Returns whether single_host mode is enabled."""
    return _single_host_mode_enabled
