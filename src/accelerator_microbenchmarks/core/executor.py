"""Execution strategies for standard and soaking JAX benchmarks."""

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import enum
import os
import time
import traceback
from typing import Any, Optional

from accelerator_microbenchmarks.core import config as config_module
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import throttle_profiler
import jax
from jax.experimental import multihost_utils
import numpy as np

# Subdirectory names under `local_xprof_dir` for soaking trace snapshots:
# `baseline_window` captures the post-warmup baseline at t=0; `worst_window`
# captures the post-soak throttled state immediately after Phase 2 soak.
_BASELINE_TRACE_WINDOW_SUBDIR = "baseline_window"
_WORST_TRACE_WINDOW_SUBDIR = "worst_window"

# Trace-reduction `ProfileOptions.advanced_configuration` applied to every
# traced benchmark loop. These drop profiler events that no TPUMS parser
# consumes, keeping large traces (e.g. multi-GiB all_reduce) under the
# profiler / trace viewer event limit so every traced run is captured.
_BASE_PROFILER_ADVANCED_CONFIG: dict[str, Any] = {
    # Already the XProf default; pinned explicitly so it cannot drift.
    "tpu_trace_mode": "TRACE_ONLY_XLA",
    # Timing events (MARKER, `call-done`, `jit_*`) are recorded on TensorCore
    # planes, while SparseCore planes dominate the event count for
    # SparseCore-offloaded collectives.
    "tpu_num_sparse_cores_to_trace": 0,
    "tpu_num_sparse_core_tiles_to_trace": 0,
    # Hardware PMU counters are unused (firmware thermal telemetry used by
    # soaking benchmarks is recorded separately).
    "tpu_perf_counters": False,
}


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
  """Outcome of an executor's measurement loop.

  Attributes:
    durations_ms: Per-iteration (or per-chunk for soaking) wall-clock latency.
    actual_runs: Total iterations executed by the host loop.
    xprof_expected_runs: Number of traced iterations whose XProf durations feed
      the `xprof_*` latency stats (the upper bound for `xprof_captured_runs`),
      or None when the loop was not traced.
  """

  durations_ms: list[float]
  actual_runs: int
  xprof_expected_runs: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class XprofDevice:
  """A TPU device identified by its host process and local hardware ID."""

  process_index: int
  local_hardware_id: int

  @property
  def label(self) -> str:
    """Returns the `detailed.json` device key `process_{p}/TPU:{id}`."""
    return f"process_{self.process_index}/TPU:{self.local_hardware_id}"


@dataclasses.dataclass(frozen=True)
class XprofParseResult:
  """XProf step durations parsed from one benchmark trace.

  Attributes:
    durations: Aggregated per-step durations (ms) that feed the headline
      `xprof_*` metrics.
    measured_device: Device whose series is `durations`, or None when
      `durations` is empty, aggregates several devices
      (`GLOBAL_SLOWEST_PER_STEP`), or measures `/host:CPU`.
    per_local_device_durations: Per-step durations (ms) of each
      `get_devices_to_measure()` device on this host, or None for `/host:CPU`
      timing, single-owner multihost benchmarks, or when any such device has no
      XProf durations.
  """

  durations: list[float]
  measured_device: Optional[XprofDevice] = None
  per_local_device_durations: Optional[dict[XprofDevice, list[float]]] = None


def _make_parse_result(
    durations: list[float],
    measured_device: Optional[XprofDevice] = None,
    per_local_device_durations: Optional[dict[XprofDevice, list[float]]] = None,
) -> XprofParseResult:
  """Builds an `XprofParseResult` that never names a device for no data.

  Args:
    durations: Headline per-step durations (ms).
    measured_device: Device behind `durations`; dropped if `durations` is empty.
    per_local_device_durations: Per-step durations of each measured local
      device; dropped if any device (or the whole map) has no durations.

  Returns:
    The parse result.
  """
  if per_local_device_durations is not None and (
      not per_local_device_durations
      or not all(per_local_device_durations.values())
  ):
    per_local_device_durations = None
  return XprofParseResult(
      durations=durations,
      measured_device=measured_device if durations else None,
      per_local_device_durations=per_local_device_durations,
  )


class _XprofParseStatus(enum.IntEnum):
  OK = 0
  PARSE_FAILED = 1
  UNEQUAL_LOCAL_COUNTS = 2


class BaseRunExecutor:
  """Standard execution strategy for JAX microbenchmarks."""

  def __init__(self, benchmark: Any):
    self.benchmark = benchmark
    self.config = benchmark.config

  def run_warmup(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Executes warmup iterations to trigger JIT compilation."""
    warmup_start = time.perf_counter()
    i = 0
    while i < self.config.warmup_tries or (
        time.perf_counter() - warmup_start
        < min(1.0, self.config.min_duration_s / 5)
    ):
      inputs = self.benchmark.reset_data(*inputs)
      outputs = self.benchmark.run_op(*inputs)
      jax.block_until_ready(outputs)
      i += 1
    return inputs

  def _get_profiler_options(self) -> jax.profiler.ProfileOptions:
    """Returns ProfileOptions that drop trace events no TPUMS parser uses."""
    options = jax.profiler.ProfileOptions()
    options.advanced_configuration = dict(_BASE_PROFILER_ADVANCED_CONFIG)
    return options

  def _get_trace_kwargs(self) -> dict[str, Any]:
    """Returns `jax.profiler.trace` kwargs wrapping `_get_profiler_options()`."""
    return {
        "create_perfetto_link": False,
        "profiler_options": self._get_profiler_options(),
    }

  def execute(
      self,
      inputs: tuple[Any, ...],
      local_xprof_dir: Optional[str],
  ) -> ExecutionResult:
    """Executes the standard single-step measurement loop."""
    ctx = contextlib.nullcontext()
    is_traced = (
        self.benchmark.xprof_config.xprof_timing and local_xprof_dir is not None
    )
    if is_traced:
      ctx = jax.profiler.trace(local_xprof_dir, **self._get_trace_kwargs())

    raw_times: list[float] = []
    actual_runs = 0
    loop_start = time.perf_counter()

    with ctx:
      while actual_runs < self.config.num_runs or (
          time.perf_counter() - loop_start < self.config.min_duration_s
      ):
        inputs = self.benchmark.reset_data(*inputs)
        t0 = time.perf_counter()
        outputs = self.benchmark.run_op(*inputs)
        jax.block_until_ready(outputs)
        t1 = time.perf_counter()
        actual_runs += 1
        if actual_runs < 1000:
          raw_times.append((t1 - t0) * 1000.0)
        # Untimed: after this run's timer stops, before the next one starts.
        self.benchmark.post_run_processing(outputs, run_index=actual_runs - 1)

    # Every measured iteration runs inside the trace, so all `actual_runs`
    # should appear in it; fewer captured means the trace was truncated.
    return ExecutionResult(
        durations_ms=raw_times,
        actual_runs=actual_runs,
        xprof_expected_runs=actual_runs if is_traced else None,
    )

  def _load_trace(self, trace_dir: str) -> Optional[profiler.XprofTrace]:
    """Loads the XProf trace in `trace_dir` once for per-device parsing."""
    return profiler.load_xprof_trace(
        trace_dir, self.benchmark.match_xprof_op_fallback
    )

  def _durations_for_device(
      self,
      trace: Optional[profiler.XprofTrace],
      local_device_id: int | None,
  ) -> list[float]:
    """Extracts XProf durations from `trace` for `local_device_id` (or `/host:CPU` if None).

    Args:
      trace: Trace loaded by `_load_trace`, or None if no trace or timing events
        were found.
      local_device_id: Local hardware ID of the target TPU device
        (`/device:TPU:{id}`), or `None` to extract `/host:CPU` events.

    Returns:
      List of extracted event durations in milliseconds.
    """
    if trace is None:
      return []
    return trace.device_step_durations_ms(local_device_id=local_device_id)

  def _local_device_durations(
      self,
      trace: Optional[profiler.XprofTrace],
      devices: Sequence[XprofDevice],
  ) -> dict[XprofDevice, list[float]]:
    """Returns per-step durations of each device in `devices`."""
    per_device: dict[XprofDevice, list[float]] = {}
    for device in devices:
      try:
        per_device[device] = self._durations_for_device(
            trace, device.local_hardware_id
        )
      except ValueError:
        raise
      except Exception as e:
        raise RuntimeError(
            f"XProf parsing failed for {device.label}: {e}"
        ) from e
    return per_device

  def _xprof_devices_to_measure(self) -> list[XprofDevice]:
    """Returns `benchmark.get_devices_to_measure()` as local `XprofDevice`s.

    Raises:
      ValueError: If a returned device is not a local device of this process or
        has no `local_hardware_id`.
    """
    raw_devices = self.benchmark.get_devices_to_measure()
    local_devices = jax.local_devices()
    if remote := [d for d in raw_devices if d not in local_devices]:
      raise ValueError(
          f"get_devices_to_measure() returned devices {remote} that are not"
          f" local to process {jax.process_index()}."
      )
    process_index = jax.process_index()
    devices: list[XprofDevice] = []
    for dev in raw_devices:
      if dev.local_hardware_id is None:
        raise ValueError(f"Device {dev} has no local_hardware_id.")
      devices.append(XprofDevice(process_index, dev.local_hardware_id))
    return devices

  @staticmethod
  def _slowest_device(
      per_device: dict[XprofDevice, list[float]],
  ) -> Optional[XprofDevice]:
    """Returns the device with the highest median duration (first wins on ties)."""
    best_device: Optional[XprofDevice] = None
    best_median = -1.0
    for device, durations in per_device.items():
      if durations:
        med = float(np.median(durations))
        if med > best_median:
          best_median = med
          best_device = device
    return best_device

  def _extract_local_device_slices(
      self,
      trace_dir: str,
  ) -> tuple[_XprofParseStatus, dict[XprofDevice, list[float]]]:
    """Extracts per-device XProf duration lists and status code across local devices.

    Args:
      trace_dir: Directory containing XProf trace files.

    Returns:
      A tuple `(status, device_slices)` where `device_slices` maps each local
      device to its per-step durations when `status` is
      `_XprofParseStatus.OK`, or is `{}` otherwise.
    """
    if not trace_dir:
      return _XprofParseStatus.PARSE_FAILED, {}

    # Validate outside the broad except below: a non-local device or missing
    # local_hardware_id is a benchmark bug that must raise, not become
    # PARSE_FAILED.
    devices = self._xprof_devices_to_measure()
    try:
      device_slices = self._local_device_durations(
          self._load_trace(trace_dir), devices
      )
    except Exception:  # pylint: disable=broad-exception-caught
      print(
          "Warning: Local XProf parsing failed in multihost sync:\n"
          f"{traceback.format_exc()}"
      )
      return _XprofParseStatus.PARSE_FAILED, {}

    if not device_slices or any(
        not device_slice for device_slice in device_slices.values()
    ):
      return _XprofParseStatus.PARSE_FAILED, {}

    if len({len(device_slice) for device_slice in device_slices.values()}) > 1:
      return _XprofParseStatus.UNEQUAL_LOCAL_COUNTS, {}

    return _XprofParseStatus.OK, device_slices

  def _parse_multihost_durations(
      self,
      trace_dir: str,
      agg_fn: Callable[..., Any] = np.max,
  ) -> XprofParseResult:
    """Parses and aggregates XProf durations across local devices and hosts.

    Uses a two-stage `multihost_utils.process_allgather` protocol outside the
    traced region:
      1. Gather `(status, n_steps)` (`int32[2]`) across all hosts so every host
         makes the identical collective decision (raise `ValueError` on unequal
         per-device or per-host step counts, or return `[]` if any host failed
         local XProf parsing).
      2. When all hosts are `OK` with equal `n_steps > 0`, apply `agg_fn`
         across local devices (`axis=0`), gather the per-host step vectors
         (`float32[n_steps]`), and apply `agg_fn` across hosts (`axis=0`).

    Args:
      trace_dir: Directory containing XProf trace files.
      agg_fn: Reduction operator accepting `(array, axis=0)` (default:
        `np.max`).

    Returns:
      Aggregated per-step durations across all local devices and hosts (empty
      if any host failed local XProf parsing) with this host's per-device
      durations.

    Raises:
      ValueError: If any host has unequal per-device step counts or if step
        counts differ across hosts.
    """
    status, device_slices = self._extract_local_device_slices(trace_dir)
    n_steps = (
        len(next(iter(device_slices.values())))
        if status == _XprofParseStatus.OK
        else 0
    )
    local_meta = np.array([status, n_steps], dtype=np.int32)
    gathered_meta = np.asarray(
        multihost_utils.process_allgather(local_meta), dtype=np.int32
    ).reshape(-1, 2)

    statuses = gathered_meta[:, 0]
    counts = gathered_meta[:, 1]
    ok_counts = counts[statuses == _XprofParseStatus.OK]

    if np.any(statuses == _XprofParseStatus.UNEQUAL_LOCAL_COUNTS) or (
        len(set(ok_counts.tolist())) > 1
    ):
      raise ValueError(
          "Unequal XProf step counts across devices or hosts:"
          f" statuses={statuses.tolist()}, step_counts={counts.tolist()}"
      )

    if np.any(statuses != _XprofParseStatus.OK) or n_steps <= 0:
      return _make_parse_result(durations=[])

    local_agg = np.asarray(
        agg_fn(
            np.asarray(list(device_slices.values()), dtype=np.float32), axis=0
        ),
        dtype=np.float32,
    )
    gathered_steps = np.asarray(
        multihost_utils.process_allgather(local_agg), dtype=np.float64
    ).reshape(-1, n_steps)
    return _make_parse_result(
        durations=[float(x) for x in agg_fn(gathered_steps, axis=0)],
        per_local_device_durations=device_slices,
    )

  def _parse_multihost_owner_durations(
      self, trace_dir: str
  ) -> XprofParseResult:
    """Parses XProf durations on the owner host and broadcasts them to all hosts."""
    measured_device = self.benchmark.get_device_to_measure()
    is_owner = jax.process_index() == measured_device.process_index
    print(
        f"[Host {jax.process_index()}] Measured device:"
        f" {measured_device.id}(p{measured_device.process_index}),"
        f" owner_process={measured_device.process_index}, is_owner={is_owner}"
    )
    # Non-addressable devices report local_hardware_id=None, so the owner
    # broadcasts its ID together with the step count. A missing owner ID is
    # sent as -1 and raised after the broadcast so no host hangs in it.
    owner_hw_id = (
        measured_device.local_hardware_id
        if is_owner and measured_device.local_hardware_id is not None
        else -1
    )
    local_durations: list[float] = []
    if is_owner:
      if owner_hw_id >= 0 and trace_dir:
        try:
          local_durations = self._durations_for_device(
              self._load_trace(trace_dir), measured_device.local_hardware_id
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          print(f"Error parsing local XProf trace: {e}")
    else:
      print(
          f"Note: Measured device {measured_device.id} is on remote Host"
          f" {measured_device.process_index} (current Host is"
          f" {jax.process_index()}). Skipping local XProf parsing."
      )

    try:
      meta = np.asarray(
          multihost_utils.broadcast_one_to_all(
              np.array([len(local_durations), owner_hw_id], dtype=np.int32),
              is_source=is_owner,
          )
      ).reshape(-1)
      n_steps = int(meta[0])
      owner_device_id = int(meta[1])
      if owner_device_id < 0:
        raise ValueError(
            f"Device {measured_device} on owner process"
            f" {measured_device.process_index} has no local_hardware_id."
        )
      measured = XprofDevice(measured_device.process_index, owner_device_id)
      if n_steps <= 0:
        return _make_parse_result(durations=[], measured_device=measured)
      local_arr = (
          np.asarray(local_durations, dtype=np.float32)
          if is_owner
          else np.zeros(n_steps, dtype=np.float32)
      )
      synced = multihost_utils.broadcast_one_to_all(
          local_arr, is_source=is_owner
      )
      return _make_parse_result(
          durations=[
              float(x) for x in np.asarray(synced, dtype=np.float64).reshape(-1)
          ],
          measured_device=measured,
      )
    except ValueError:
      raise
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: multihost broadcast failed ({e}); using local values.")
      # Only the owner has local durations; the non-owner's are dropped.
      return _make_parse_result(
          durations=local_durations,
          measured_device=(
              XprofDevice(measured_device.process_index, owner_hw_id)
              if is_owner
              else None
          ),
      )

  def _parse_xprof_durations(
      self,
      trace_dir: str,
  ) -> XprofParseResult:
    """Extracts XProf device durations from trace_dir according to XprofDeviceTimingAggregation.

    Each branch loads the trace at most once and reuses it for every device.

    Args:
      trace_dir: Directory containing XProf trace files.

    Returns:
      The parsed durations, measured device, and per-local-device durations.
    """
    if self.benchmark.xprof_target_host_cpu:
      return _make_parse_result(
          durations=self._durations_for_device(
              self._load_trace(trace_dir), None
          )
      )

    aggregation = self.benchmark.xprof_config.device_timing_aggregation
    if (
        aggregation
        == constants.XprofDeviceTimingAggregation.GLOBAL_SLOWEST_PER_STEP
    ):
      return self._parse_multihost_durations(
          trace_dir,
          agg_fn=np.max,
      )

    if self.benchmark.requires_multihost_sync:
      return self._parse_multihost_owner_durations(trace_dir)

    trace = self._load_trace(trace_dir)
    if (
        aggregation
        == constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE
    ):
      per_device = self._local_device_durations(
          trace, self._xprof_devices_to_measure()
      )
      slowest = self._slowest_device(per_device)
      return _make_parse_result(
          durations=per_device[slowest] if slowest is not None else [],
          measured_device=slowest,
          per_local_device_durations=per_device,
      )

    # Without `requires_multihost_sync`, the measured device is local.
    measured_device = self.benchmark.get_device_to_measure()
    return self._parse_single_device(
        trace,
        XprofDevice(jax.process_index(), measured_device.local_hardware_id),
    )

  def _parse_single_device(
      self,
      trace: Optional[profiler.XprofTrace],
      measured: XprofDevice,
  ) -> XprofParseResult:
    """Returns `measured`'s durations, plus per-local-device ones when collected.

    The headline series never depends on parsing other devices: a failure on
    another device only drops `per_local_device_durations` (logged).

    Args:
      trace: Trace loaded by `_load_trace`, or None.
      measured: Device behind the headline series.

    Returns:
      The parse result.

    Raises:
      ValueError: If `measured` is not one of `get_devices_to_measure()`.
    """
    devices = self._xprof_devices_to_measure()
    if measured not in devices:
      raise ValueError(
          f"Measured device {measured.label} is not in"
          f" get_devices_to_measure() ({[d.label for d in devices]})."
      )
    durations = self._durations_for_device(trace, measured.local_hardware_id)
    per_device: dict[XprofDevice, list[float]] = {}
    for device in devices:
      if device == measured:
        per_device[device] = durations
        continue
      try:
        per_device[device] = self._durations_for_device(
            trace, device.local_hardware_id
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(
            f"Warning: Per-local-device XProf parsing failed for {device.label}"
            f" ({e}); metrics_per_local_device will be null."
        )
        return _make_parse_result(durations=durations, measured_device=measured)
    return _make_parse_result(
        durations=durations,
        measured_device=measured,
        per_local_device_durations=per_device,
    )

  def get_xprof_durations(self, xprof_dir: str) -> XprofParseResult:
    """Returns XProf kernel durations (ms) for the completed benchmark run."""
    return self._parse_xprof_durations(xprof_dir)

  def _compute_basic_latency_stats(
      self, times_ms: Sequence[float], prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Computes basic latency statistics (avg, p50, p90, std) without filtering."""
    if not times_ms:
      return {
          f"{prefix}_avg_ms": 0.0,
          f"{prefix}_p50_ms": 0.0,
          f"{prefix}_p90_ms": 0.0,
          f"{prefix}_std_ms": 0.0,
      }
    return {
        f"{prefix}_p50_ms": float(np.percentile(times_ms, 50)),
        f"{prefix}_p90_ms": float(np.percentile(times_ms, 90)),
        f"{prefix}_avg_ms": float(np.mean(times_ms)),
        f"{prefix}_std_ms": float(np.std(times_ms)),
    }

  def calculate_latency_stats(
      self, times_ms: list[float], prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Derives statistical latency metrics with 1.5*IQR outlier filtering."""
    if len(times_ms) > 3:
      q1 = np.percentile(times_ms, 25)
      q3 = np.percentile(times_ms, 75)
      iqr = q3 - q1
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr
      filtered_times = [t for t in times_ms if lower_bound <= t <= upper_bound]
    else:
      filtered_times = times_ms

    return self._compute_basic_latency_stats(filtered_times, prefix)

  def augment_host_metrics(self, metrics: dict[str, Any]) -> None:
    """Hook to attach execution-strategy-specific host/thermal metrics."""
    del metrics

  def augment_xprof_metrics(
      self,
      metrics: dict[str, Any],
  ) -> None:
    """Hook to attach execution-strategy-specific XProf metrics."""
    del metrics


class SoakingRunExecutor(BaseRunExecutor):
  """Continuous soaking execution strategy with baseline, untraced soak, and post-soak worst window.

  Chronological Execution Pipeline:
    1. Warmup (`run_warmup`): Executes untraced chunks (up to `samples_per_run`
       iterations per chunk) until `warmup_tries` iterations have elapsed.
    2. Phase 1 — Post-Warmup Baseline (`baseline_window` at `t = 0s`): Executes
       1 traced window of `samples_per_run` iterations immediately after warmup
       to capture post-warmup baseline device latency
       (`xprof_pre_soaking_p50_ms`).
    3. Phase 2 — Untraced Thermal Soak (`min_duration_s` > 0.0): Executes
       continuous, untraced chunks of `samples_per_run` iterations until elapsed
       Phase 2 time `>= min_duration_s` to drive the accelerator to thermal
       steady state without `jax.profiler.trace` flush gaps (`num_runs` is not
       used in soaking mode). Wall-clock soaking metrics
       (`wall_clock_pre_soaking_p50_ms`, `wall_clock_post_soaking_p50_ms`,
       `wall_clock_slowdown_ratio`) are derived exclusively from the first and
       last untraced soak chunks (`soak_wall_ms[0]` and `soak_wall_ms[-1]`).
    4. Phase 3 — Post-Soak Worst Window (`worst_window` at `t = end of soak`):
       Executes 1 traced window (`worst_window`) of `samples_per_run` iterations
       immediately after Phase 2 soak to capture post-soak device latency
       (`xprof_post_soaking_p50_ms`) and firmware thermal telemetry.
  """

  def __init__(self, benchmark: Any):
    super().__init__(benchmark)
    self.first_window_xprof_duration: list[float] = []
    self.last_window_xprof_duration: list[float] = []
    # Per-chunk wall-clock latency (ms/iter) of the untraced Phase 2 soak
    # chunks; the traced Phase 1/3 windows are excluded.
    self.soak_wall_ms: list[float] = []
    self._result_details: dict[str, Any] = {}

  def _samples_per_run(self) -> int:
    """Returns the number of back-to-back iterations per timing window."""
    return int(self.config.samples_per_run)

  def _get_profiler_options(self) -> jax.profiler.ProfileOptions:
    """Extends the base trace-reduction options with TPU FW thermal/power events."""
    options = super()._get_profiler_options()
    options.host_tracer_level = 0
    options.advanced_configuration = {
        **options.advanced_configuration,
        "tpu_e2e_enable_fw_thermal_event": True,
        "tpu_e2e_enable_fw_power_level_event": True,
        "tpu_e2e_enable_fw_throttle_event": True,
        "tpu_power_trace_level": 1,
    }
    return options

  def _run_chunk(self, *inputs, samples: int = 1) -> tuple[jax.Array, ...]:
    """Executes `samples` back-to-back iterations and blocks once at the end."""
    outputs = None
    for _ in range(samples):
      inputs = self.benchmark.reset_data(*inputs)
      outputs = self.benchmark.run_op(*inputs)
    if outputs is not None:
      jax.block_until_ready(outputs)
    return inputs

  def _run_window(
      self,
      inputs: tuple[Any, ...],
      samples: int,
      trace_dir: Optional[str] = None,
      **trace_kwargs: Any,
  ) -> tuple[tuple[Any, ...], float]:
    """Executes `samples` iterations (optionally traced) and returns (inputs, per_iter_ms)."""
    ctx = (
        jax.profiler.trace(trace_dir, **trace_kwargs)
        if trace_dir is not None
        else contextlib.nullcontext()
    )
    with ctx:
      t0 = time.perf_counter()
      inputs = self._run_chunk(*inputs, samples=samples)
      t1 = time.perf_counter()
    per_iter_ms = ((t1 - t0) * 1000.0) / samples
    return inputs, per_iter_ms

  def run_warmup(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Executes chunked warmup iterations (honoring exact `warmup_tries`) before soaking measurement."""
    samples_per_run = self._samples_per_run()
    warmup_tries = int(self.config.warmup_tries)
    i = 0
    while i < warmup_tries:
      step_iters = min(samples_per_run, warmup_tries - i)
      inputs = self._run_chunk(*inputs, samples=step_iters)
      i += step_iters
    return inputs

  def _extract_window_durations(
      self,
      window_trace_dir: str,
  ) -> XprofParseResult:
    """Pure helper that extracts XProf durations from `window_trace_dir`."""
    try:
      return self._parse_xprof_durations(
          window_trace_dir,
      )
    except ValueError:
      raise
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"Warning: Failed to parse XProf durations for {window_trace_dir}:"
          f" {e}"
      )
      return XprofParseResult(durations=[])

  def execute(
      self,
      inputs: tuple[Any, ...],
      local_xprof_dir: Optional[str],
  ) -> ExecutionResult:
    """Executes Phase 1 (`baseline_window`), Phase 2 (`min_duration_s` soak), and Phase 3 (`worst_window`)."""
    if not self.benchmark.xprof_config.xprof_timing or not local_xprof_dir:
      raise ValueError(
          "SoakingRunExecutor requires xprof_timing=True and a valid"
          " local_xprof_dir."
      )
    if self.config.min_duration_s <= 0.0:
      raise ValueError("SoakingRunExecutor requires min_duration_s > 0.0.")

    samples_per_run = self._samples_per_run()

    trace_kwargs = self._get_trace_kwargs()

    raw_times: list[float] = []
    self.soak_wall_ms = []
    actual_runs = 0

    # Phase 1: Post-warmup baseline window at t=0 before soaking begins.
    baseline_dir = os.path.join(local_xprof_dir, _BASELINE_TRACE_WINDOW_SUBDIR)
    inputs, baseline_ms = self._run_window(
        inputs,
        samples_per_run,
        trace_dir=baseline_dir,
        **trace_kwargs,
    )
    raw_times.append(baseline_ms)
    actual_runs += samples_per_run

    # Phase 2: Continuous untraced thermal soak (`min_duration_s`)
    # (keeps MXU at 100% duty cycle with zero XProf teardown overhead).
    soak_start = time.perf_counter()
    while True:
      inputs, soak_ms = self._run_window(inputs, samples_per_run)
      self.soak_wall_ms.append(soak_ms)
      actual_runs += samples_per_run
      if (time.perf_counter() - soak_start) >= self.config.min_duration_s:
        break

    # Phase 3: Post-soak worst window immediately after Phase 2 soak.
    worst_dir = os.path.join(local_xprof_dir, _WORST_TRACE_WINDOW_SUBDIR)
    inputs, worst_ms = self._run_window(
        inputs,
        samples_per_run,
        trace_dir=worst_dir,
        **trace_kwargs,
    )
    raw_times.extend(self.soak_wall_ms)
    raw_times.append(worst_ms)
    actual_runs += samples_per_run
    print(
        "SoakingRunExecutor wall-clock ms/iter:"
        f" baseline_window={baseline_ms:.4f},"
        f" soak_p50={np.percentile(self.soak_wall_ms, 50):.4f}"
        f" ({len(self.soak_wall_ms)} chunks), worst_window={worst_ms:.4f}"
    )

    # xprof_* stats use only the traced worst_window (`samples_per_run`
    # iterations); baseline_window and the untraced soak are excluded.
    return ExecutionResult(
        durations_ms=raw_times,
        actual_runs=actual_runs,
        xprof_expected_runs=samples_per_run,
    )

  def get_xprof_durations(self, xprof_dir: str) -> XprofParseResult:
    """Parses both trace windows; returns the post-soak worst_window result.

    Args:
      xprof_dir: Directory containing the `baseline_window` and `worst_window`
        trace subdirectories.

    Returns:
      The worst_window result (durations, measured device, and
      per-local-device durations), which feeds the headline `xprof_*` metrics
      and `detailed.json`; the baseline_window only feeds
      `xprof_pre_soaking_p50_ms`.
    """
    baseline_dir = os.path.join(xprof_dir, _BASELINE_TRACE_WINDOW_SUBDIR)
    worst_dir = os.path.join(xprof_dir, _WORST_TRACE_WINDOW_SUBDIR)
    if (
        self.benchmark.xprof_config.device_timing_aggregation
        == constants.XprofDeviceTimingAggregation.LOCAL_SLOWEST_DEVICE
    ):
      worst_result = self._extract_window_durations(worst_dir)
      if worst_result.measured_device is None:
        self.first_window_xprof_duration = self._extract_window_durations(
            baseline_dir
        ).durations
      else:
        try:
          self.first_window_xprof_duration = self._durations_for_device(
              self._load_trace(baseline_dir),
              worst_result.measured_device.local_hardware_id,
          )
        except ValueError:
          raise
        except Exception as e:  # pylint: disable=broad-exception-caught
          print(
              f"Warning: Failed to parse XProf durations for {baseline_dir}:"
              f" {e}"
          )
          self.first_window_xprof_duration = []
    else:
      self.first_window_xprof_duration = self._extract_window_durations(
          baseline_dir
      ).durations
      worst_result = self._extract_window_durations(worst_dir)

    self.last_window_xprof_duration = worst_result.durations
    return worst_result

  def calculate_latency_stats(
      self, times_ms: list[float], prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    """Derives latency stats without IQR outlier removal, including phase-aligned slowdown ratio."""
    stats = self._compute_basic_latency_stats(times_ms, prefix)
    if not times_ms:
      for suffix in constants.SOAKING_LATENCY_SUFFIXES:
        stats[f"{prefix}_{suffix}"] = None
      return stats

    # Wall-clock pre_soaking/post_soaking windows are from the first and last
    # untraced Phase 2 soak chunks (each averaging `samples_per_run` iterations,
    # matching XProf's first-window vs last-window definition). The traced
    # Phase 1/3 windows are excluded because `worst_window` can read
    # un-throttled on fast clock-skip throttling hosts, whereas the soak chunks
    # reflect the throttled steady state. Falls back to `times_ms` when no soak
    # samples were recorded.
    if prefix == constants.TimingDomain.WALL_CLOCK and self.soak_wall_ms:
      phase_times = self.soak_wall_ms
    else:
      phase_times = times_ms
    pre_soaking_p50_ms = float(phase_times[0])
    post_soaking_p50_ms = float(phase_times[-1])

    slowdown_ratio = (
        post_soaking_p50_ms / pre_soaking_p50_ms
        if pre_soaking_p50_ms > 0
        else 1.0
    )
    stats.update({
        f"{prefix}_{constants.PRE_SOAKING_P50_MS}": pre_soaking_p50_ms,
        f"{prefix}_{constants.POST_SOAKING_P50_MS}": post_soaking_p50_ms,
        f"{prefix}_{constants.SLOWDOWN_RATIO}": slowdown_ratio,
    })
    return stats

  def augment_host_metrics(self, metrics: dict[str, Any]) -> None:
    """Merges per-window firmware telemetry."""
    local_xprof_dir = self.benchmark._xprof_dir_actual or ""  # pylint: disable=protected-access

    baseline_dir = ""
    worst_dir = ""
    if local_xprof_dir:
      baseline_dir = os.path.join(
          local_xprof_dir, _BASELINE_TRACE_WINDOW_SUBDIR
      )
      worst_dir = os.path.join(local_xprof_dir, _WORST_TRACE_WINDOW_SUBDIR)

    thermal_metrics, self._result_details = (
        throttle_profiler.XprofThermalProfiler.parse_soaking_windows(
            baseline_dir=baseline_dir,
            worst_dir=worst_dir,
        )
    )
    metrics.update(thermal_metrics)

  def augment_xprof_metrics(
      self,
      metrics: dict[str, Any],
  ) -> None:
    """Computes soaking XProf latency (`pre_soaking_p50_ms`, `post_soaking_p50_ms`, `slowdown_ratio`).

    Each window is reported independently, so a window without XProf
    durations does not hide the other one; the slowdown ratio needs both.

    Args:
      metrics: Metrics dictionary updated in place.
    """

    def window_p50(durations: Sequence[float]) -> Optional[float]:
      if not durations:
        return None
      p50 = float(np.percentile(durations, 50))
      return p50 if p50 > 0 else None

    pre_soak_p50 = window_p50(self.first_window_xprof_duration)
    post_soak_p50 = window_p50(self.last_window_xprof_duration)
    ratio = None
    if pre_soak_p50 is not None and post_soak_p50 is not None:
      ratio = post_soak_p50 / pre_soak_p50 if pre_soak_p50 > 0 else 1.0
    metrics[constants.XPROF_PRE_SOAKING_P50_MS] = pre_soak_p50
    metrics[constants.XPROF_POST_SOAKING_P50_MS] = post_soak_p50
    metrics[constants.XPROF_SLOWDOWN_RATIO] = ratio

  def get_result_details(self) -> dict[str, Any]:
    """Returns per-device telemetry of both trace windows."""
    return dict(self._result_details)


def create_executor(benchmark: Any) -> BaseRunExecutor:
  """Factory returning SoakingRunExecutor or BaseRunExecutor for `benchmark`."""
  if isinstance(benchmark.config, config_module.SoakingExecutionParamsMixin):
    return SoakingRunExecutor(benchmark)
  return BaseRunExecutor(benchmark)
