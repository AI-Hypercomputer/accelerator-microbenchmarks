"""Roofline analysis for benchmark metrics."""

from typing import Any, Dict, Optional


def apply_roofline_analysis(
    benchmark_instance, metrics: Dict[str, Any], **params
) -> Dict[str, Any]:
  """Apply roofline estimation to finalized metrics."""
  use_trace = params.get("use_trace_roofline", False)

  if use_trace:
    # Override intensity if tracing is requested
    trace_stats = benchmark_instance._get_trace_metrics(**params)
    if trace_stats:
      metrics["trace_flops"] = trace_stats.get("flops", 0)
      metrics["trace_hbm_bytes"] = trace_stats.get("hbm_bytes", 0)
      # Recalculate intensity from trace
      if metrics["trace_hbm_bytes"] > 0:
        metrics["intensity"] = (
            metrics["trace_flops"] / metrics["trace_hbm_bytes"]
        )

  if "hardware_stats" in params:
    hw = params["hardware_stats"]
    if isinstance(hw, dict):
      dtype = params.get("dtype", "bfloat16")
      tflops_dict = hw.get("tflops", {})
      if isinstance(tflops_dict, dict):
        peak_tflops = tflops_dict.get(dtype, tflops_dict.get("bfloat16", 0.0))
      else:
        peak_tflops = float(tflops_dict) if tflops_dict else 0.0
      hbm_bw_data = hw.get("hbm_bw", 0.0)
    else:
      peak_tflops = 0.0
      hbm_bw_data = 0.0

    intensity = benchmark_instance.get_arithmetic_intensity(**params)
    total_bytes = benchmark_instance.get_total_bytes(**params)

    # 1. Resolve BW for this transfer size
    if isinstance(hbm_bw_data, (int, float)):
      bw = hbm_bw_data
    elif isinstance(hbm_bw_data, list):
      sorted_data = sorted(hbm_bw_data, key=lambda x: x[0])
      if not sorted_data:
        bw = 0.0
      elif total_bytes <= sorted_data[0][0]:
        bw = sorted_data[0][1]
      elif total_bytes >= sorted_data[-1][0]:
        bw = sorted_data[-1][1]
      else:
        bw = 0.0
        for i in range(len(sorted_data) - 1):
          s0, bw0 = sorted_data[i]
          s1, bw1 = sorted_data[i + 1]
          if s0 <= total_bytes <= s1:
            bw = bw0 + (bw1 - bw0) * (total_bytes - s0) / (s1 - s0)
            break
    elif isinstance(hbm_bw_data, dict):
      sorted_sizes = sorted([int(k) for k in hbm_bw_data.keys()])
      if not sorted_sizes:
        bw = 0.0
      elif total_bytes <= sorted_sizes[0]:
        bw = (
            hbm_bw_data[str(sorted_sizes[0])]
            if str(sorted_sizes[0]) in hbm_bw_data
            else hbm_bw_data[sorted_sizes[0]]
        )
      elif total_bytes >= sorted_sizes[-1]:
        bw = (
            hbm_bw_data[str(sorted_sizes[-1])]
            if str(sorted_sizes[-1]) in hbm_bw_data
            else hbm_bw_data[sorted_sizes[-1]]
        )
      else:
        bw = 0.0
        for i in range(len(sorted_sizes) - 1):
          s0, s1 = sorted_sizes[i], sorted_sizes[i + 1]
          if s0 <= total_bytes <= s1:
            v0 = (
                hbm_bw_data[str(s0)]
                if str(s0) in hbm_bw_data
                else hbm_bw_data[s0]
            )
            v1 = (
                hbm_bw_data[str(s1)]
                if str(s1) in hbm_bw_data
                else hbm_bw_data[s1]
            )
            bw = v0 + (v1 - v0) * (total_bytes - s0) / (s1 - s0)
            break
    else:
      bw = 0.0

    # 2. Compute Roofline
    roofline_tflops = min(peak_tflops, (intensity * bw) / 1000.0)
    metrics["roofline_tflops_limit"] = roofline_tflops
    metrics["peak_bw_at_size_gb_s"] = bw

    # 3. Efficiency
    actual_tflops = metrics.get("tflops_per_sec", 0.0)
    if actual_tflops > 0 and roofline_tflops > 0:
      metrics["roofline_efficiency"] = (actual_tflops / roofline_tflops) * 100.0

    # 4. Bandwidth Efficiency (for memory bound ops)
    actual_bw = metrics.get("bandwidth_gb_s", 0.0)
    if actual_bw > 0 and bw > 0:
      metrics["bw_efficiency"] = (actual_bw / bw) * 100.0

  return metrics
