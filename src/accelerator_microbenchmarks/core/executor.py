"""Execution strategies for standard and soaking JAX benchmarks."""

from collections.abc import Sequence
import contextlib
import os
import time
from typing import Any, Optional

from accelerator_microbenchmarks.core import config as config_module
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import profiler
from accelerator_microbenchmarks.core import throttle_profiler
import jax
import numpy as np

# Subdirectory names under `local_xprof_dir` for soaking trace snapshots:
# `baseline_window` captures the post-warmup baseline at t=0; `worst_window`
# captures the post-soak throttled state immediately after Phase 2 soak.
_BASELINE_TRACE_WINDOW_SUBDIR = "baseline_window"
_WORST_TRACE_WINDOW_SUBDIR = "worst_window"


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

  def execute(
      self,
      inputs: tuple[Any, ...],
      local_xprof_dir: Optional[str],
  ) -> tuple[list[float], int]:
    """Executes the standard single-step measurement loop."""
    ctx = contextlib.nullcontext()
    if self.benchmark.xprof_config.xprof_timing and local_xprof_dir is not None:
      ctx = jax.profiler.trace(local_xprof_dir, create_perfetto_link=False)

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

    return raw_times, actual_runs

  def _resolve_max_device_id(self, trace_dir: str) -> Optional[int]:
    """Selects the local device ID with the highest median kernel duration."""
    if self.benchmark.xprof_target_host_cpu:
      return None
    best_dev_id: Optional[int] = None
    best_median = -1.0
    for dev in jax.local_devices():
      dev_id = dev.local_hardware_id
      durs = profiler.parse_xprof_durations(
          trace_dir,
          self.benchmark.match_xprof_op_fallback,
          local_device_id=dev_id,
      )
      if durs:
        med = float(np.median(durs))
        if med > best_median:
          best_median = med
          best_dev_id = dev_id
    return best_dev_id

  def _parse_xprof_durations(
      self,
      trace_dir: str,
      target_device_id: Optional[int] = None,
  ) -> list[float]:
    """Extracts XProf device durations from trace_dir according to XprofDeviceMode."""

    def _durations_for_device(dev_id: Optional[int]) -> list[float]:
      return profiler.parse_xprof_durations(
          trace_dir,
          self.benchmark.match_xprof_op_fallback,
          local_device_id=dev_id,
      )

    if self.benchmark.xprof_target_host_cpu:
      return _durations_for_device(None)

    if target_device_id is not None:
      return _durations_for_device(target_device_id)

    if (
        self.benchmark.xprof_config.device_mode
        == constants.XprofDeviceMode.MAX_DEVICE
    ):
      device_slices: list[list[float]] = []
      for dev in jax.local_devices():
        dev_slice = _durations_for_device(dev.local_hardware_id)
        if dev_slice:
          device_slices.append(dev_slice)
      return (
          max(device_slices, key=lambda s: float(np.median(s)))
          if device_slices
          else []
      )

    measured_device = self.benchmark.get_device_to_measure()
    return _durations_for_device(measured_device.local_hardware_id)

  def get_xprof_durations(self, xprof_dir: str) -> list[float]:
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
      has_xprof_timings: bool = True,
  ) -> None:
    """Hook to attach execution-strategy-specific XProf metrics."""
    del metrics, has_xprof_timings


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
    self.xprof_durations: list[float] = []
    self.first_window_xprof_duration: list[float] = []
    self.last_window_xprof_duration: list[float] = []
    # Per-chunk wall-clock latency (ms/iter) of the untraced Phase 2 soak
    # chunks; the traced Phase 1/3 windows are excluded.
    self.soak_wall_ms: list[float] = []
    self._target_device_id: Optional[int] = None

  def _samples_per_run(self) -> int:
    """Returns the number of back-to-back iterations per timing window."""
    return int(self.config.samples_per_run)

  def _get_profiler_options(self) -> Any:
    """Returns ProfileOptions enabling TPU FW thermal/power events."""
    try:
      options = jax.profiler.ProfileOptions()
      options.host_tracer_level = 0
      options.advanced_configuration = {
          "tpu_e2e_enable_fw_thermal_event": True,
          "tpu_e2e_enable_fw_power_level_event": True,
          "tpu_e2e_enable_fw_throttle_event": True,
          "tpu_power_trace_level": 1,
          "tpu_perf_counters": False,
      }
      return options
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: Failed to configure jax.profiler.ProfileOptions: {e}")
      return None

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
      self, window_trace_dir: str, target_device_id: Optional[int] = None
  ) -> list[float]:
    """Pure helper that extracts XProf durations from `window_trace_dir`."""
    try:
      return self._parse_xprof_durations(
          window_trace_dir,
          target_device_id=target_device_id,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"Warning: Failed to parse XProf durations for {window_trace_dir}:"
          f" {e}"
      )
      return []

  def execute(
      self,
      inputs: tuple[Any, ...],
      local_xprof_dir: Optional[str],
  ) -> tuple[list[float], int]:
    """Executes Phase 1 (`baseline_window`), Phase 2 (`min_duration_s` soak), and Phase 3 (`worst_window`)."""
    if not self.benchmark.xprof_config.xprof_timing or not local_xprof_dir:
      raise ValueError(
          "SoakingRunExecutor requires xprof_timing=True and a valid"
          " local_xprof_dir."
      )
    if self.config.min_duration_s <= 0.0:
      raise ValueError("SoakingRunExecutor requires min_duration_s > 0.0.")

    samples_per_run = self._samples_per_run()

    trace_kwargs: dict[str, Any] = {"create_perfetto_link": False}
    profile_options = self._get_profiler_options()
    if profile_options is not None:
      trace_kwargs["profiler_options"] = profile_options

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

    self._target_device_id = None
    if (
        self.benchmark.xprof_config.device_mode
        == constants.XprofDeviceMode.MAX_DEVICE
    ):
      self._target_device_id = self._resolve_max_device_id(worst_dir)

    return raw_times, actual_runs

  def get_xprof_durations(self, xprof_dir: str) -> list[float]:
    """Returns aggregated baseline and worst-window XProf durations."""
    baseline_dir = os.path.join(xprof_dir, _BASELINE_TRACE_WINDOW_SUBDIR)
    worst_dir = os.path.join(xprof_dir, _WORST_TRACE_WINDOW_SUBDIR)
    self.first_window_xprof_duration = self._extract_window_durations(
        baseline_dir, target_device_id=self._target_device_id
    )
    self.last_window_xprof_duration = self._extract_window_durations(
        worst_dir, target_device_id=self._target_device_id
    )
    self.xprof_durations = list(self.last_window_xprof_duration)
    return list(self.xprof_durations)

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
      has_xprof_timings: bool = True,
  ) -> None:
    """Computes soaking XProf latency (`pre_soaking_p50_ms`, `post_soaking_p50_ms`, `slowdown_ratio`).

    Each window is reported independently, so a window without XProf
    durations does not hide the other one; the slowdown ratio needs both.

    Args:
      metrics: Metrics dictionary updated in place.
      has_xprof_timings: Whether XProf durations are usable at all.
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
    return dict(getattr(self, "_result_details", {}))


def create_executor(benchmark: Any) -> BaseRunExecutor:
  """Factory returning SoakingRunExecutor or BaseRunExecutor for `benchmark`."""
  if isinstance(benchmark.config, config_module.SoakingExecutionParamsMixin):
    return SoakingRunExecutor(benchmark)
  return BaseRunExecutor(benchmark)
