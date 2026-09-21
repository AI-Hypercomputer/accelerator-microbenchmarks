# Developer Guide: Adding New Benchmarks

This guide explains how to extend `accelerator-microbenchmarks` (`tpums`) with
new accelerator operations or composite microbenchmarks.

## 1. Benchmark Anatomy

Every benchmark must inherit from `BaseBenchmark[TConfig]`, define a typed
parameter dataclass (`Config`), and register itself via
`@registry.benchmark_registry.register(...)` under
`src/accelerator_microbenchmarks/benchmarks/`.

```python
import dataclasses
from typing import Any, Callable, Sequence

from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import utils
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class MyNewOpParams(base.SingleDtypeBenchmarkParams):
  size: int = dataclasses.field(
      default=1048576,
      metadata={"min": 1, "help": "Number of elements in the input tensor."},
  )


@registry.benchmark_registry.register("my_new_op")
class MyNewOpBenchmark(base.BaseBenchmark[MyNewOpParams]):
  """Microbenchmark for a custom elementwise operation."""

  Config = MyNewOpParams
  roofline_mode = constants.RooflineMode.MEMORY_HBM
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("size", report.format_str),
      ("total_bytes_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_chip_gb_s", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_chip_gb_s", report.format_2f),
  )

  def setup(self) -> None:
    """Called once before warmup to JIT-compile the kernel."""

    @jax.jit
    def my_op(x):
      with jax.named_scope(constants.MARKER):
        return x * 2.0

    self._jit_fn = my_op

  def get_run_identifier(self) -> str:
    return f"size_{self.config.size}_{self.config.dtype}"

  def generate_inputs(self) -> tuple[jax.Array, ...]:
    """Allocates input tensors on device and returns arguments for run_op."""
    dtype = utils.parse_dtype(self.config.dtype)
    x = jnp.ones((self.config.size,), dtype=dtype)
    return (x,)

  def run_op(self, *args, **kwargs) -> Any:
    """Executes the compiled kernel inside warmup and measurement loops."""
    return self._jit_fn(*args, **kwargs)
```

## 2. Throughput Metrics & Roofline Analysis

`BaseBenchmark` requires subclasses to implement three abstract methods on the
benchmark class—`get_total_bytes()`, `get_arithmetic_intensity()`, and
`calculate_throughput_metrics()`—which `BaseBenchmark.calculate_metrics()`
combines with `get_workload_metadata()` and `calculate_latency_stats()` to emit
domain-prefixed throughput (`wall_clock_*` and `xprof_*`) and Roofline
efficiency metrics:

```python
  def get_total_bytes(self) -> float:
    """Calculates total bytes moved to/from HBM (1 read + 1 write)."""
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize
    return float(self.config.size * itemsize * 2)

  def get_arithmetic_intensity(self) -> float:
    """Calculates arithmetic intensity (FLOPs per byte moved)."""
    flops = float(self.config.size)
    return flops / self.get_total_bytes()

  def get_workload_metadata(self) -> dict[str, Any]:
    return {
        "total_bytes_mib": self.get_total_bytes() / (1024 * 1024),
        "intensity": self.get_arithmetic_intensity(),
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    latency_s = latency_ms / 1000.0
    bw_gb_s = (
        (self.get_total_bytes() / latency_s) / 1e9
        if latency_s > 0
        else float("inf")
    )
    return {
        f"{prefix}_bandwidth_per_device_gb_s": bw_gb_s,
    }
```

When `derive_chip_bandwidth = True` (the default on `BaseBenchmark`),
`BaseBenchmark.derive_chip_metrics()` automatically computes
`*_bandwidth_per_chip_gb_s` and `*_tflops_per_chip` from per-device metrics
using `self.hardware_spec.devices_per_chip`.

## 3. Registering & Discovering the Benchmark

1.  **Module Discovery**: Import your new module inside
    `load_all_benchmarks()` in
    `src/accelerator_microbenchmarks/benchmarks/benchmark_loader.py` so the CLI
    and runner automatically register it at startup.
2.  **Compiler Flags (`op_flags.yaml`)**: Add an entry for `my_new_op` in
    `src/accelerator_microbenchmarks/op_flags.yaml` (or map it in
    `_BENCHMARK_NAME_MAPPING` in `core/runner.py`) to configure hardware-optimal
    `LIBTPU_INIT_ARGS`.

## 4. Best Practices

-   **Typed Dataclass Configs**: Inherit from `SingleDtypeBenchmarkParams` when
    your benchmark uses a single `dtype` parameter, or `BaseBenchmarkParams`
    when using separate `in_dtype` / `out_dtype` fields. Use `metadata={"min":
    ..., "help": ...}` on dataclass fields for automatic CLI `--help` generation
    and bounds validation.
-   **XProf Marker Scope**: Wrap the core kernel inside
    `with jax.named_scope(constants.MARKER):` so XProf trace parsing accurately
    isolates on-device kernel duration (`xprof_p50_ms`).
-   **Mesh & Sharding Awareness**: Use `self.mesh` or explicit
    `jax.sharding.NamedSharding` / `SingleDeviceSharding` when allocating inputs
    in `generate_inputs()` to avoid host memory OOMs.
-   **Standard Units**: Report data sizes in binary mebibytes (`_mib`, divided
    by `1024 * 1024`) and bandwidths in decimal gigabytes per second (`_gb_s`,
    divided by `1e9`).

## 5. Local Verification

Verify your new benchmark via the `tpums` CLI and unit test suite:

```bash
# Inspect auto-generated CLI flags from your Config dataclass
tpums benchmark run my_new_op --help

# Execute interactively with XProf hardware timing
tpums benchmark run my_new_op --xprof_timing --size 1048576 --dtype bfloat16
```
