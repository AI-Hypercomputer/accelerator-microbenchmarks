"""Base class for all JAX benchmarks."""

import abc
import dataclasses
import datetime
import enum
import functools
import numbers
import os
import time
import types
from typing import Any, Callable, ClassVar, Generic, Mapping, Optional, Sequence, TypeVar

from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.core import executor
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import roofline
from accelerator_microbenchmarks.core import system
import jax
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class XprofConfig:
  """Configuration for XProf trace collection and device timing analysis."""

  xprof_timing: bool = False
  xprof_dir: str = "/tmp/tensorboard"
  device_timing_aggregation: str = (
      constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST
  )


# Field-metadata key that `dtype_field()` sets to mark a dtype field.
_DTYPE_FIELD_KEY = "dtype"


def dtype_field(
    *, default: dtypes.DType | str, metadata: Mapping[str, Any] | None = None
) -> Any:
  """Returns a dataclass field holding a `dtypes.DType` (e.g. 'bfloat16').

  `BaseBenchmarkParams` validates such fields and coerces them into `DType`
  members (see `BaseBenchmarkParams.allowed_dtypes()`).

  Args:
    default: The default dtype, as a `dtypes.DType` member or its value.
    metadata: Additional field metadata, e.g. `{"help": ...}`.

  Returns:
    A `dataclasses.Field` marked as a dtype field.
  """
  return dataclasses.field(
      default=default,
      metadata={**(metadata or {}), _DTYPE_FIELD_KEY: True},
  )


def is_dtype_field(field: dataclasses.Field[Any]) -> bool:
  """Returns whether `field` was declared with `dtype_field()`."""
  return bool(field.metadata.get(_DTYPE_FIELD_KEY, False))


@dataclasses.dataclass
class BaseBenchmarkParams:
  """Base configuration parameters for microbenchmarks."""

  # Allowed dtypes per dtype field (declared with `dtype_field()`), as jnp
  # scalar types (e.g. `jnp.bfloat16`) or abstract dtype classes (e.g.
  # `jnp.floating`). Dtype fields that are not listed accept every
  # `dtypes.DType` member. Every key must name a dtype field and match at least
  # one `dtypes.DType` member. Subclasses replace (rather than merge)
  # `DTYPE_CONSTRAINTS`; merge explicitly via
  # `{**Parent.DTYPE_CONSTRAINTS, ...}` to preserve inherited constraints.
  DTYPE_CONSTRAINTS: ClassVar[Mapping[str, Sequence[jax.typing.DTypeLike]]] = {}
  # Set False where bytes are counted with `itemsize`: TPUs pack sub-byte
  # dtypes (e.g. `int4`), so that would over-report bandwidth.
  ALLOW_SUB_BYTE_DTYPES: ClassVar[bool] = True
  _explicit_fields: ClassVar[frozenset[str]] = frozenset()

  def __new__(cls, *args: Any, **kwargs: Any):
    instance = super().__new__(cls)
    init_fields = [f.name for f in dataclasses.fields(cls) if f.init]
    explicit = set(init_fields[: len(args)]) | (set(kwargs) & set(init_fields))
    object.__setattr__(instance, "_explicit_fields", frozenset(explicit))
    return instance

  warmup_tries: int = dataclasses.field(
      default=10,
      metadata={
          "min": 0,
          "help": "Number of warmup iterations before measurement.",
      },
  )
  num_runs: int = dataclasses.field(
      default=10,
      metadata={"min": 0, "help": "Number of measurement iterations."},
  )
  min_duration_s: float = dataclasses.field(
      default=0.0,
      metadata={
          "min": 0.0,
          "help": (
              "Minimum measurement duration in seconds. Also runs warmup for"
              " at least min_duration_s / 5 (capped at 1.0s)."
          ),
      },
  )

  def _validate_bounds(self) -> None:
    """Checks every field against its min/max metadata constraints.

    Raises:
      ValueError: If any field violates its `min`/`max` bound or integer type.
    """
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      minimum = field.metadata.get("min")
      maximum = field.metadata.get("max")

      if field.type is int:
        # bool is an Integral subclass, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
          bound_str = f" >= {minimum}" if minimum is not None else ""
          raise ValueError(
              f"{field.name} must be an integer{bound_str}, got {value!r}"
          )
        # Normalize integer-like values (e.g. numpy ints) so bounds apply.
        value = int(value)
        setattr(self, field.name, value)
      elif value is None:
        continue
      elif not isinstance(value, (int, float)) or isinstance(value, bool):
        continue

      if minimum is not None and value < minimum:
        raise ValueError(f"{field.name} must be >= {minimum}, got {value}")

      if maximum is not None and value > maximum:
        raise ValueError(f"{field.name} must be <= {maximum}, got {value}")

  def _coerce_enums(self) -> None:
    """Coerces every enum field in-place into a member of its Enum type.

    Raises:
      ValueError: If any field is not a valid member of its Enum type.
    """
    for field in dataclasses.fields(self):
      # Dtype fields are coerced into `dtypes.DType` by `_validate_dtypes()`,
      # whose errors list the dtypes the field accepts.
      if is_dtype_field(field):
        continue
      # Checks if the field is an enum by coercing raw values into their Enum
      # member. Only scalar enum fields are supported, not containers such as
      # list[Enum].
      if isinstance(field.type, type) and issubclass(field.type, enum.Enum):
        enum_type: type[enum.Enum] = field.type
      else:
        continue

      value = getattr(self, field.name)
      try:
        setattr(self, field.name, enum_type(value))
      except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid {field.name} '{value}'. Must be one of"
            f" {[member.value for member in enum_type]}."
        ) from exc

  @classmethod
  def allowed_dtypes(cls, field_name: str) -> tuple[dtypes.DType, ...]:
    """Returns the `dtypes.DType` members that dtype field `field_name` accepts.

    Args:
      field_name: The name of a dtype field of this class.

    Returns:
      The members that `_dtype_rejection_reasons()` accepts, in `DType` order:
      those matching the field's `DTYPE_CONSTRAINTS` entry (all if it has
      none), without sub-byte members unless `ALLOW_SUB_BYTE_DTYPES` is set.

    Raises:
      ValueError: If `field_name` is not a dtype field.
      TypeError: If the field's `DTYPE_CONSTRAINTS` entry is invalid (see
        `dtypes.dtypes_matching()`) or leaves no allowed member. Raising here
        rather than returning `()` also fails `cli.py`, which uses this for
        argparse `choices`, at parser construction.
    """
    if not any(
        f.name == field_name and is_dtype_field(f)
        for f in dataclasses.fields(cls)
    ):
      raise ValueError(f"{cls.__name__}.{field_name} is not a dtype field.")
    members = tuple(
        member
        for member, reason in cls._dtype_rejection_reasons(field_name).items()
        if reason is None
    )
    if not members:
      raise TypeError(
          f"{cls.__name__}.DTYPE_CONSTRAINTS[{field_name!r}] matches no"
          " allowed DType member (sub-byte members count only if"
          " ALLOW_SUB_BYTE_DTYPES is set)."
      )
    return members

  @classmethod
  @functools.cache
  def _dtype_rejection_reasons(
      cls, field_name: str
  ) -> Mapping[dtypes.DType, str | None]:
    """Returns why `field_name` rejects each `DType` member (None: accepts).

    Checks `DTYPE_CONSTRAINTS`, then `ALLOW_SUB_BYTE_DTYPES` (x64 is checked in
    `_to_allowed_dtype()`). Cached per (class, field), so don't patch these
    after first use.

    Args:
      field_name: The name of a dtype field of this class.

    Raises:
      TypeError: If the field's `DTYPE_CONSTRAINTS` entry is invalid.
    """
    matching = None  # No entry: every member matches.
    if field_name in cls.DTYPE_CONSTRAINTS:
      constraints = cls.DTYPE_CONSTRAINTS[field_name]
      # Unlike `dtypes_matching()`, an empty entry matches nothing.
      matching = (
          frozenset(dtypes.dtypes_matching(*constraints))
          if constraints
          else frozenset()
      )
    reasons = {}
    for member in dtypes.DType:
      if matching is not None and member not in matching:
        reasons[member] = f"Unsupported dtype '{member.value}'."
      # Pass the np.dtype rather than the name: `dtypes.parse_dtype()` would
      # reject 64-bit names while `jax_enable_x64` is off.
      elif not cls.ALLOW_SUB_BYTE_DTYPES and dtypes.is_sub_byte_dtype(
          member.jnp_dtype
      ):
        reasons[member] = (
            f"Unsupported sub-byte dtype '{member.value}'. TPUs store it"
            f" packed below one byte per element, but {cls.__name__} counts"
            " itemsize (1 byte), which over-reports bandwidth."
        )
      else:
        reasons[member] = None
    return types.MappingProxyType(reasons)

  def _validate_dtypes(self) -> None:
    """Validates every dtype field and coerces it into a `DType` member.

    Raises:
      TypeError: If `DTYPE_CONSTRAINTS` is misconfigured.
      ValueError: If a dtype field's value is invalid or unsupported.
    """
    self._check_dtype_config()
    for field in dataclasses.fields(self):
      if is_dtype_field(field):
        value = getattr(self, field.name)
        setattr(self, field.name, self._to_allowed_dtype(field.name, value))

  def _check_dtype_config(self) -> None:
    """Raises TypeError if `DTYPE_CONSTRAINTS` doesn't fit the dtype fields.

    That is, if a key is not a dtype field, or a dtype field allows no `DType`.
    """
    dtype_fields = [
        f.name for f in dataclasses.fields(self) if is_dtype_field(f)
    ]
    unknown = set(self.DTYPE_CONSTRAINTS) - set(dtype_fields)
    if unknown:
      raise TypeError(
          f"{type(self).__name__}.DTYPE_CONSTRAINTS names non-dtype fields:"
          f" {sorted(unknown)}."
      )
    for name in dtype_fields:
      self.allowed_dtypes(name)  # Raises TypeError if no member is allowed.

  def _to_allowed_dtype(self, field_name: str, value: Any) -> dtypes.DType:
    """Returns `value` as a `DType` member allowed for `field_name`.

    Checks, in order (ValueError on the first failure):
      1. `value` is a `DType` value.
      2. `dtypes.parse_dtype` accepts it. Runs before 3 so a 64-bit dtype
         reports the `jax_enable_x64` fix rather than "unsupported".
      3. `_dtype_rejection_reasons()` accepts it for the field.

    Args:
      field_name: The name of a dtype field.
      value: The field's value.
    """
    member = dtypes.DType.from_value(value)
    if member is None:
      raise self._dtype_error(field_name, f"Invalid dtype {value!r}.")
    try:
      dtypes.parse_dtype(member.value)
    except ValueError as exc:
      raise ValueError(f"{field_name}: {exc}") from exc
    reason = self._dtype_rejection_reasons(field_name)[member]
    if reason is not None:
      raise self._dtype_error(field_name, reason)
    return member

  def _dtype_error(self, field_name: str, reason: str) -> ValueError:
    """Returns a ValueError for `field_name` that lists its allowed dtypes."""
    allowed = ", ".join(self.allowed_dtypes(field_name))
    return ValueError(
        f"{field_name}: {reason} Supported dtypes for {field_name}: {allowed}"
    )

  def __post_init__(self):
    self._coerce_enums()
    self._validate_bounds()
    self._validate_dtypes()

  def with_generation_defaults(
      self, param_defaults: dict[str, Any]
  ) -> "BaseBenchmarkParams":
    """Returns a copy with `param_defaults` applied to non-explicit fields.

    Must be called before `expand_test_cases()`, because `dataclasses.replace()`
    and `dataclasses.asdict()` copies pass every field to `__init__` and mark
    all fields as explicit.

    Args:
      param_defaults: Mapping of field names to generation-specific default
        values.
    """
    if not param_defaults:
      return self
    explicit_fields = self._explicit_fields
    init_fields = {f.name for f in dataclasses.fields(self) if f.init}
    updates = {
        k: v
        for k, v in param_defaults.items()
        if k in init_fields and k not in explicit_fields
    }
    if not updates:
      return self
    updated = dataclasses.replace(self, **updates)
    object.__setattr__(updated, "_explicit_fields", explicit_fields)
    return updated

  def expand_test_cases(self) -> Sequence["BaseBenchmarkParams"]:
    """Default 1-to-1 mapping: returns [self]."""
    return [self]


@dataclasses.dataclass
class SingleDtypeBenchmarkParams(BaseBenchmarkParams):
  """Base parameters for benchmarks using a single tensor data type."""

  dtype: dtypes.DType = dtype_field(
      default=dtypes.DType.BFLOAT16,
      metadata={
          "help": "Data type for tensor operations (e.g. bfloat16, float32)."
      },
  )


@dataclasses.dataclass
class ScopeParamsMixin:
  """Mixin for benchmarks targeting a host-local device, chip, or host scope."""

  scope: constants.Scope = dataclasses.field(
      default=constants.Scope.DEVICE,
      metadata={
          "help": (
              "Target measurement scope: 'device' (single local device),"
              " 'chip' (all devices on one local chip), or 'host' (all local"
              " devices)."
          ),
      },
  )
  local_index: int = dataclasses.field(
      default=0,
      metadata={
          "min": 0,
          "help": (
              "Host-local index of the target unit within `scope`: the N-th"
              " local device (scope=device) or N-th local chip (scope=chip)."
              " Must be 0 for scope=host."
          ),
      },
  )


@dataclasses.dataclass
class BenchmarkMetadata:
  """Metadata for a single benchmark run."""

  benchmark_name: str
  test_name: str
  start_time: str
  end_time: str
  params: dict[str, Any]
  platform_info: platform.PlatformInfo
  hardware_spec: system.HardwareSpec
  xla_flags: str = ""
  libtpu_init_args: str = ""
  xprof_config: Optional[XprofConfig] = None


@dataclasses.dataclass
class WallClockTimings:
  """Host wall-clock timings of a benchmark run."""

  durations_ms: list[float]


@dataclasses.dataclass
class XprofTimings:
  """XProf hardware timings of a benchmark run."""

  durations_ms: list[float]
  measured_device: Optional[str] = None
  metrics_per_local_device: Optional[dict[str, dict[str, float]]] = None


@dataclasses.dataclass
class BenchmarkTimings:
  """All timings for a benchmark run."""

  wall_clock: WallClockTimings
  xprof: Optional[XprofTimings] = None


@dataclasses.dataclass
class BenchmarkResult:
  """Results of a benchmark run."""

  metadata: BenchmarkMetadata
  metrics: dict[str, Any]
  timings: BenchmarkTimings


TConfig = TypeVar("TConfig", bound=BaseBenchmarkParams)


class BaseBenchmark(Generic[TConfig], abc.ABC):
  """Abstract base class for microbenchmarks."""

  name: str = ""
  Config = BaseBenchmarkParams  # pylint: disable=invalid-name
  is_experimental: bool = False
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = ()
  # TODO(b/555967269): Redesign and decouple circular dependency between
  # BaseBenchmark and report formatters.
  REPORT_FORMATTERS: Optional[
      Sequence[Callable[[pd.DataFrame, type["BaseBenchmark"]], str]]
  ] = None
  derive_chip_bandwidth: bool = True
  roofline_mode: constants.RooflineMode = constants.RooflineMode.NONE
  # Whether the benchmark needs inter-device (ICI) communication. Fail-closed
  # default: benchmarks that keep all data movement on the host<->device path
  # opt out by setting this to False, which allows them to run under
  # `standalone_host_init` (no SliceBuilder, hence no ICI).
  requires_ici: bool = True

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if "name" not in cls.__dict__:
      cls.name = cls.__name__

  def __init__(
      self,
      config: TConfig,
      hardware_spec: system.HardwareSpec,
      mesh: Optional[jax.sharding.Mesh] = None,
      xprof_config: Optional[XprofConfig] = None,
  ):
    self.config: TConfig = config
    self.hardware_spec: system.HardwareSpec = hardware_spec
    self.mesh: Optional[jax.sharding.Mesh] = mesh
    self.xprof_config: XprofConfig = xprof_config or XprofConfig()
    self._jit_fn = None
    self._xprof_dir_actual: str = self.xprof_config.xprof_dir
    self._xprof_dir_cns: str = self._xprof_dir_actual
    self.executor: executor.BaseRunExecutor = executor.create_executor(self)

  def _create_default_mesh(self) -> jax.sharding.Mesh:
    """Create a default 1D mesh spanning all available devices."""
    devices = jax.devices()
    return jax.sharding.Mesh(np.array(devices), axis_names=("device",))

  @abc.abstractmethod
  def run_op(self, *args, **kwargs) -> Any:
    """The core operation intended for performance assessment."""

  def setup(self):
    """Perform setup such as JIT compilation or buffer pre-allocation."""
    # Mesh creation is deferred to the run method.
    pass

  def teardown(self) -> None:
    """Release resources acquired in setup() (e.g. pinned host buffers).

    Called by run() after every run, including when the run raises, so
    resources held by one sweep point are released before the next sweep
    point's benchmark instance calls setup(). Implementations must be
    idempotent and safe to call before setup().
    """
    pass

  def reset_data(self, *inputs, **kwargs) -> tuple[jax.Array, ...]:
    """Reset data for the next run."""
    return inputs

  def post_run_processing(self, outputs: Any, run_index: int) -> None:
    """Process the outputs of each timed run or window, outside the timer.

    Called by BaseRunExecutor (inside the profiler trace, so it must not emit
    MARKER named scopes and must synchronize any device work it dispatches
    before returning) and by WindowedRunExecutor subclasses (outside the
    profiler trace), never in warmup.

    Args:
      outputs: Outputs returned by the timed run's (or window's final) `run_op`.
      run_index: Zero-based index of the timed run or window.
    """
    del outputs, run_index  # Unused in base class.

  def get_run_identifier(self) -> str:
    """Return a string identifier for the current run parameters."""
    return ""

  @abc.abstractmethod
  def generate_inputs(self) -> tuple[Any, ...]:
    """Generate or retrieve inputs for the benchmark.

    Returns:
      A tuple of arguments passed to run_op.
    """
    pass

  def get_arithmetic_intensity(self) -> float:
    """Return operational intensity (FLOPs/Byte) for roofline analysis."""
    raise NotImplementedError(
        f"{self.__class__.__name__} must implement get_arithmetic_intensity() "
        "when roofline_mode is COMPUTE or MEMORY_HBM."
    )

  def get_peak_ici_bandwidth_per_chip(self) -> float | None:
    """Return the theoretical peak ICI bus bandwidth per chip (GB/s).

    Required when roofline_mode == RooflineMode.NETWORK_ICI. Returns None
    when the peak cannot be modeled for the current configuration.
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} must implement"
        " get_peak_ici_bandwidth_per_chip() when roofline_mode =="
        " RooflineMode.NETWORK_ICI."
    )

  def get_peak_link_bandwidth(
      self,
  ) -> tuple[float, constants.TransferLink] | None:
    """Return the peak link bandwidth per device (GB/s) and the bounding link.

    Required when roofline_mode == RooflineMode.NETWORK_LINK. Returns None
    when the peak cannot be modeled for the current configuration.
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} must implement"
        " get_peak_link_bandwidth() when roofline_mode =="
        " RooflineMode.NETWORK_LINK."
    )

  @abc.abstractmethod
  def get_total_bytes(self) -> float:
    """Calculate total bytes moved to/from HBM."""
    pass

  def calculate_metrics(self, times_ms: list[float]) -> dict[str, Any]:
    """Derive static workload metadata, host wall-clock latency statistics, and domain throughput metrics."""
    metrics = self.get_workload_metadata()
    if self.roofline_mode in (
        constants.RooflineMode.COMPUTE,
        constants.RooflineMode.MEMORY_HBM,
    ):
      metrics["intensity"] = self.get_arithmetic_intensity()
    latency_stats = self.calculate_latency_stats(
        times_ms, prefix=constants.TimingDomain.WALL_CLOCK
    )
    metrics.update(latency_stats)
    metrics.update(
        self.calculate_throughput_metrics(
            latency_ms=latency_stats[
                f"{constants.TimingDomain.WALL_CLOCK}_p50_ms"
            ],
            prefix=constants.TimingDomain.WALL_CLOCK,
        )
    )
    self.executor.augment_host_metrics(metrics)
    return metrics

  def get_workload_metadata(self) -> dict[str, Any]:
    """Return static, timing-invariant workload metadata (e.g., total_flops, data_size_mib)."""
    return {}

  def calculate_latency_stats(
      self, times_ms: list[float], prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Derive statistical latency metrics (avg, p50, p90, std) from raw timing data."""
    return self.executor.calculate_latency_stats(times_ms, prefix=prefix)

  @abc.abstractmethod
  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Calculate domain-prefixed throughput metrics from a representative latency (ms)."""

  def match_xprof_op_fallback(self, event: dict[str, Any]) -> bool:
    """Fallback to capture relevant xprof op when MARKER is not present."""
    del event  # Unused in base class.
    return False

  def apply_roofline_analysis(self, metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply roofline estimation to finalized metrics."""

    return roofline.apply_roofline_analysis(self, metrics)

  def get_compute_dtype(self) -> str:
    """Return the primary data type used for compute math, to determine peak TFLOPS."""
    if not hasattr(self.config, "dtype"):
      raise ValueError(
          f"Benchmark config {self.config} does not have a 'dtype' attribute. "
          "Please implement get_compute_dtype() in the subclass."
      )
    return self.config.dtype

  def get_device_to_measure(self) -> jax.Device:
    """Returns the local JAX Device to observe in XProf.

    Defaults to the first device in `get_devices_to_measure()` (local device 0
    unless a subclass narrows the participating devices).
    """
    return self.get_devices_to_measure()[0]

  def get_devices_to_measure(self) -> list[jax.Device]:
    """Returns the local JAX Devices the workload runs on (default: all).

    XProf per-device durations (`timings.xprof.metrics_per_local_device`,
    `local_slowest_device`, and the per-host stage of
    `global_slowest_per_step`) are collected over exactly these devices, in
    this order. Every device must be local to this process. Scope-parameterized
    benchmarks (`HBMBandwidthBenchmark`, `_HostDeviceBenchmark`) return
    `self.get_devices_for_scope()`. This is not used for `xprof_target_host_cpu`
    or `requires_multihost_sync` benchmarks.
    """
    return list(jax.local_devices())

  def get_xprof_device_timing_aggregation(self) -> str:
    """Returns the effective XProf device timing aggregation mode."""
    return self.xprof_config.device_timing_aggregation

  @staticmethod
  def _group_devices_by_chip(
      local_devices: Sequence[jax.Device],
      devices_per_chip: int,
  ) -> list[list[jax.Device]]:
    """Groups `local_devices` into chips in encounter order."""
    if devices_per_chip == 1:
      return [[device] for device in local_devices]
    chips_by_coords: dict[tuple[int, ...], list[jax.Device]] = {}
    for device in local_devices:
      coords = tuple(device.coords)
      chips_by_coords.setdefault(coords, []).append(device)
    for coords, chip_devices in chips_by_coords.items():
      if len(chip_devices) != devices_per_chip:
        raise ValueError(
            f"Chip at coords={coords} has {len(chip_devices)} local device(s),"
            f" expected devices_per_chip={devices_per_chip}."
        )
    return list(chips_by_coords.values())

  def get_devices_for_scope(self) -> list[jax.Device]:
    """Resolves local JAX devices for `self.config.scope` and `local_index`."""
    scope = self.config.scope  # pyrefly: ignore[missing-attribute]
    idx = self.config.local_index  # pyrefly: ignore[missing-attribute]
    local_devices = list(jax.local_devices())
    if not local_devices:
      raise ValueError("No local JAX devices available.")

    if scope == constants.Scope.HOST:
      if idx != 0:
        raise ValueError(f"local_index must be 0 when scope='host', got {idx}.")
      return local_devices
    if scope == constants.Scope.DEVICE:
      if idx >= len(local_devices):
        raise ValueError(
            f"local_index {idx} out of range for scope='device' with"
            f" {len(local_devices)} local device(s) (valid:"
            f" 0..{len(local_devices) - 1})."
        )
      return [local_devices[idx]]
    if scope == constants.Scope.CHIP:
      if self.hardware_spec is None:
        raise ValueError(
            "scope='chip' requires a non-None hardware_spec to determine"
            " devices_per_chip."
        )
      chips = self._group_devices_by_chip(
          local_devices, self.hardware_spec.devices_per_chip
      )
      if idx >= len(chips):
        raise ValueError(
            f"local_index {idx} out of range for scope='chip' with"
            f" {len(chips)} local chip(s) (valid: 0..{len(chips) - 1})."
        )
      return chips[idx]
    raise ValueError(f"Unsupported scope: {scope!r}")

  def scope_bandwidth_metrics(
      self,
      prefix: constants.TimingDomain | str,
      bandwidth_gb_s: float | None,
  ) -> dict[str, float | None]:
    """Builds `{prefix}_bandwidth_per_{device,chip,host}_gb_s` for `self.config.scope`.

    Populates the active `scope` with `bandwidth_gb_s`, derives smaller scopes
    as the average share of the concurrent aggregate measurement, and leaves
    larger scopes as `None` (no upward extrapolation, since shared hardware
    bottlenecks such as a dual-die PCIe link do not scale linearly). Preserves
    `None` (all three keys `None`) and `float("inf")` on populated keys.

    Args:
      prefix: Timing domain prefix (e.g., `wall_clock` or `xprof`).
      bandwidth_gb_s: Achieved bandwidth in GB/s for the active scope.

    Returns:
      Dictionary mapping `{prefix}_bandwidth_per_{device,chip,host}_gb_s` to
      the measured bandwidth on `scope`, average-share values on smaller scopes,
      and `None` on larger scopes.
    """
    scope = self.config.scope  # pyrefly: ignore[missing-attribute]
    num_devices = len(self.get_devices_to_measure())
    if bandwidth_gb_s is None:
      per_device, per_chip, per_host = None, None, None
    else:
      per_device = bandwidth_gb_s / num_devices
      if scope == constants.Scope.DEVICE:
        per_chip, per_host = None, None
      elif scope == constants.Scope.CHIP:
        per_chip, per_host = bandwidth_gb_s, None
      elif scope == constants.Scope.HOST:
        per_chip = (
            per_device * self.hardware_spec.devices_per_chip
            if self.hardware_spec is not None
            else None
        )
        per_host = bandwidth_gb_s
      else:
        raise ValueError(f"Unsupported scope: {scope!r}")
    return {
        f"{prefix}_bandwidth_per_{constants.Scope.DEVICE.value}_gb_s": (
            per_device
        ),
        f"{prefix}_bandwidth_per_{constants.Scope.CHIP.value}_gb_s": per_chip,
        f"{prefix}_bandwidth_per_{constants.Scope.HOST.value}_gb_s": per_host,
    }

  @property
  def requires_multihost_sync(self) -> bool:
    """Whether this benchmark targets a single remote host and requires broadcast to collect metrics.

    By default (symmetric workloads like matmul, collectives, hbm), each host
    independently parses its own local XProf trace without broadcasting.
    """
    return False

  @property
  def xprof_target_host_cpu(self) -> bool:
    """Whether XProf timing measures /host:CPU events instead of /device:TPU:{id}.

    Default is False (measures accelerator hardware kernels on
    /device:TPU:{id}). _HostDeviceBenchmark overrides this to True because
    H2D/D2H DMA transfers are recorded on the CPU host process.
    """
    return False

  @property
  def supported_device_timing_aggregations(
      self,
  ) -> frozenset[constants.XprofDeviceTimingAggregation]:
    """XProf device timing aggregations this benchmark supports.

    `/host:CPU` timing has no per-device series, and `requires_multihost_sync`
    benchmarks already broadcast the owner host's series to every host, so
    both support only `measured_device_or_first`. Single-device benchmarks
    (such as `HBMBandwidthBenchmark`) override this property directly.
    Other benchmarks support every aggregation.
    """
    if self.xprof_target_host_cpu or self.requires_multihost_sync:
      return frozenset(
          {constants.XprofDeviceTimingAggregation.MEASURED_DEVICE_OR_FIRST}
      )
    return frozenset(constants.XprofDeviceTimingAggregation)

  def _validate_device_timing_aggregation(self) -> None:
    """Rejects an XProf device timing aggregation this benchmark would ignore.

    Raises:
      ValueError: If `xprof_config.device_timing_aggregation` is not in
        `supported_device_timing_aggregations`.
    """
    aggregation = self.xprof_config.device_timing_aggregation
    supported = self.supported_device_timing_aggregations
    if aggregation in supported:
      return
    if self.xprof_target_host_cpu:
      reason_suffix = (
          " (XProf timing measures /host:CPU, not per-device kernels)"
      )
    elif self.requires_multihost_sync:
      reason_suffix = (
          " (results are already synced across hosts via owner broadcast)"
      )
    else:
      reason_suffix = ""
    raise ValueError(
        f"{self.name} does not support xprof_device_timing_aggregation="
        f"{aggregation}; supported: {', '.join(sorted(supported))}"
        f"{reason_suffix}."
    )

  def _summarize_metrics_per_local_device(
      self,
      per_local_device_durations: Optional[
          dict[executor.XprofDevice, list[float]]
      ],
  ) -> Optional[dict[str, dict[str, float]]]:
    """Summarizes XProf latency per measured local device for `detailed.json`."""
    if per_local_device_durations is None:
      return None
    latency_keys = [f"{stat}_ms" for stat in ("p50", "avg", "p90", "std")]
    summary: dict[str, dict[str, float]] = {}
    for device, durations in per_local_device_durations.items():
      stats = self.calculate_latency_stats(
          durations, prefix=constants.TimingDomain.XPROF
      )
      summary[device.label] = {
          key: stats[f"{constants.TimingDomain.XPROF}_{key}"]
          for key in latency_keys
      }
    return summary

  def _apply_xprof_timing_and_sync(
      self, metrics: dict[str, Any]
  ) -> tuple[dict[str, Any], executor.XprofParseResult]:
    """Parses XProf trace, syncs across hosts, and recalculates metrics.

    Args:
      metrics: Wall-clock metrics, updated in place with `xprof_*` metrics.

    Returns:
      The updated metrics and the XProf parse result behind them (empty
      durations if local trace parsing failed).
    """
    xprof_dir = self._xprof_dir_actual
    cns_dir = self._xprof_dir_cns

    xprof_url = profiler.upload_xprof_trace(xprof_dir, cns_dir)
    if xprof_url:
      metrics["xprof_url"] = xprof_url

    all_devs_str = [f"{d.id}(p{d.process_index})" for d in jax.devices()]
    local_devs_str = [
        f"{d.id}(p{d.process_index})" for d in jax.local_devices()
    ]
    print(f"[Host {jax.process_index()}] All devices: {all_devs_str}")
    print(f"[Host {jax.process_index()}] Local devices: {local_devs_str}")
    xprof_result = executor.XprofParseResult(durations=[])
    try:
      xprof_result = self.executor.get_xprof_durations(xprof_dir)
    except ValueError:
      raise
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error parsing local XProf trace: {e}")
    durations = xprof_result.durations

    xprof_avg, xprof_p50, xprof_p90, xprof_std = 0.0, 0.0, 0.0, 0.0
    if durations:
      print(
          f"Using XProf device timings ({len(durations)} runs)"
          " to calculate derived performance metrics."
      )
      xprof_stats = self.calculate_latency_stats(
          durations, prefix=constants.TimingDomain.XPROF
      )
      xprof_avg = xprof_stats[f"{constants.TimingDomain.XPROF}_avg_ms"]
      xprof_p50 = xprof_stats[f"{constants.TimingDomain.XPROF}_p50_ms"]
      xprof_p90 = xprof_stats[f"{constants.TimingDomain.XPROF}_p90_ms"]
      xprof_std = xprof_stats[f"{constants.TimingDomain.XPROF}_std_ms"]
    else:
      print("Warning: No XProf device timings found locally.")

    metrics["xprof_captured_runs"] = len(durations)

    # Apply XProf timings or fallback to None for unmeasured durations
    if xprof_p50 > 0:
      metrics.update({
          f"{constants.TimingDomain.XPROF}_avg_ms": xprof_avg,
          f"{constants.TimingDomain.XPROF}_p50_ms": xprof_p50,
          f"{constants.TimingDomain.XPROF}_p90_ms": xprof_p90,
          f"{constants.TimingDomain.XPROF}_std_ms": xprof_std,
      })
      self.executor.augment_xprof_metrics(metrics)
      metrics.update(
          self.calculate_throughput_metrics(
              latency_ms=xprof_p50, prefix=constants.TimingDomain.XPROF
          )
      )
    else:
      print(
          "Warning: No valid XProf timings recorded; setting XProf metrics to"
          " None."
      )
      metrics.update({
          f"{constants.TimingDomain.XPROF}_avg_ms": None,
          f"{constants.TimingDomain.XPROF}_p50_ms": None,
          f"{constants.TimingDomain.XPROF}_p90_ms": None,
          f"{constants.TimingDomain.XPROF}_std_ms": None,
      })
      self.executor.augment_xprof_metrics(metrics)
      for wall_clock_key, xprof_key in (
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_device_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_device_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_chip_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_chip_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_host_gb_s",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_host_gb_s",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_host_gbps",
              f"{constants.TimingDomain.XPROF}_bandwidth_per_host_gbps",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_tflops_per_device",
              f"{constants.TimingDomain.XPROF}_tflops_per_device",
          ),
          (
              f"{constants.TimingDomain.WALL_CLOCK}_tflops_per_chip",
              f"{constants.TimingDomain.XPROF}_tflops_per_chip",
          ),
      ):
        if wall_clock_key in metrics:
          metrics[xprof_key] = None
    return metrics, xprof_result

  def derive_chip_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
    """Derives per-chip metrics from per-device metrics for available domains."""
    devices_per_chip = self.hardware_spec.devices_per_chip
    for prefix in constants.TimingDomain:
      tflops_dev_key = f"{prefix}_tflops_per_device"
      tflops_chip_key = f"{prefix}_tflops_per_chip"
      if tflops_dev_key in metrics and tflops_chip_key not in metrics:
        metrics[tflops_chip_key] = (
            metrics[tflops_dev_key] * devices_per_chip
            if metrics[tflops_dev_key] is not None
            else None
        )

      bw_dev_key = f"{prefix}_bandwidth_per_device_gb_s"
      bw_chip_key = f"{prefix}_bandwidth_per_chip_gb_s"
      if (
          self.derive_chip_bandwidth
          and bw_dev_key in metrics
          and bw_chip_key not in metrics
      ):
        metrics[bw_chip_key] = (
            metrics[bw_dev_key] * devices_per_chip
            if metrics[bw_dev_key] is not None
            else None
        )
    return metrics

  def run(self) -> BenchmarkResult:
    """Standard orchestration flow for a benchmark.

    Always calls teardown() afterwards, even if the run raises.

    Returns:
      The consolidated BenchmarkResult of the run.
    """
    try:
      return self._run()
    finally:
      self.teardown()

  def _run(self) -> BenchmarkResult:
    """Runs setup, warmup, measurement, and metric finalization."""
    if self.xprof_config and self.xprof_config.xprof_timing:
      self._validate_device_timing_aggregation()
    if self.is_experimental:
      print(
          "===================================================================="
      )
      print(
          f"[WARNING] Running experimental benchmark '{self.name}'. Metrics"
          " and API are subject to change."
      )
      print(
          "===================================================================="
      )

    # 0. Initialize mesh if not provided
    if self.mesh is None:
      self.mesh = self._create_default_mesh()

    # 1. Setup & Initial Inputs
    self.setup()
    inputs = self.generate_inputs()

    # 2. Warmup & JIT Compilation
    inputs = self.executor.run_warmup(inputs)

    benchmark_name = self.name
    if self.xprof_config.xprof_timing:
      try:
        jax.profiler.stop_trace()
      except RuntimeError:
        # No active profiler trace is running; safe to ignore.
        pass
      xprof_base_dir = self.xprof_config.xprof_dir
      timestamp = int(time.time())

      run_id = self.get_run_identifier()
      dir_suffix = f"_{run_id}" if run_id else ""

      cns_xprof_dir = os.path.join(
          xprof_base_dir, f"{benchmark_name}{dir_suffix}_{timestamp}"
      )
      local_xprof_dir = f"/tmp/microbenchmarks_tmptrace/{benchmark_name}{dir_suffix}_{timestamp}"

      if not (
          cns_xprof_dir.startswith("/cns/")
          or cns_xprof_dir.startswith("/bigstore/")
      ):
        local_xprof_dir = cns_xprof_dir

      self._xprof_dir_actual = local_xprof_dir
      self._xprof_dir_cns = cns_xprof_dir
      print(
          f"Collecting xprof trace locally to {local_xprof_dir} across runs..."
      )

    start_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 3. Measurement Loop
    loop_start = time.perf_counter()
    exec_result = self.executor.execute(
        inputs,
        self._xprof_dir_actual if self.xprof_config.xprof_timing else None,
    )

    if self.xprof_config.xprof_timing:
      print("Xprof trace collected.")

    end_ts = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    # 4. Finalize Results
    # Calculate host-side metrics
    metrics = self.calculate_metrics(exec_result.durations_ms)
    xprof_timings: Optional[XprofTimings] = None

    if self.xprof_config.xprof_timing:
      metrics, xprof_result = self._apply_xprof_timing_and_sync(metrics)
      xprof_timings = XprofTimings(
          durations_ms=xprof_result.durations,
          measured_device=(
              xprof_result.measured_device.label
              if xprof_result.measured_device is not None
              else None
          ),
          metrics_per_local_device=self._summarize_metrics_per_local_device(
              xprof_result.per_local_device_durations
          ),
      )
      expected = exec_result.xprof_expected_runs
      if expected is not None:
        metrics["xprof_expected_runs"] = expected
        captured = metrics["xprof_captured_runs"]
        if captured > expected:
          # Truncation can only lose samples; more durations than traced
          # iterations means the parser split steps incorrectly.
          raise RuntimeError(
              f"XProf parsed {captured} durations for {expected} traced runs;"
              " expected at most one duration per traced iteration. Check"
              " MARKER / match_xprof_op_fallback event matching."
          )
        if captured == 0:
          print(
              f"Warning: No XProf durations parsed for {expected} traced"
              " runs (trace parse failure, no MARKER / fallback events"
              " matched, or multihost broadcast failure)."
          )
        elif captured < expected:
          print(
              f"Warning: XProf captured {captured}/{expected} traced runs;"
              " xprof_* metrics are based on a truncated trace (profiler"
              " event limit likely exceeded)."
          )

    metrics = self.derive_chip_metrics(metrics)

    metrics = self.apply_roofline_analysis(metrics)

    metrics["total_duration_s"] = time.perf_counter() - loop_start
    metrics["actual_runs"] = exec_result.actual_runs

    effective_xprof_config: Optional[XprofConfig] = None
    if self.xprof_config.xprof_timing:
      effective_agg = self.get_xprof_device_timing_aggregation()
      if effective_agg != self.xprof_config.device_timing_aggregation:
        effective_xprof_config = dataclasses.replace(
            self.xprof_config,
            device_timing_aggregation=effective_agg,
        )
      else:
        effective_xprof_config = self.xprof_config

    metadata = BenchmarkMetadata(
        benchmark_name=benchmark_name,
        test_name=f"{benchmark_name}_{int(time.time())}",
        start_time=start_ts,
        end_time=end_ts,
        params=(
            dataclasses.asdict(self.config)
            if dataclasses.is_dataclass(self.config)
            else {}
        ),
        # TODO(simonleesyuan): Platform description is independent of benchmark
        # execution logic; move platform_info capture out of BaseBenchmark into
        # the runner orchestrator.
        platform_info=platform.get_platform_info(),
        hardware_spec=self.hardware_spec,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
        xprof_config=effective_xprof_config,
    )

    return BenchmarkResult(
        metadata=metadata,
        metrics=metrics,
        timings=BenchmarkTimings(
            wall_clock=WallClockTimings(durations_ms=exec_result.durations_ms),
            xprof=xprof_timings,
        ),
    )
