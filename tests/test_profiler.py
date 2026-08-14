"""Unit tests for profiler.py."""

import gzip
import io
import json
from unittest import mock

from absl.testing import absltest
from accelerator_microbenchmarks.core import profiler
import os


class ProfilerTest(absltest.TestCase):
  """Unit tests for profiler.py."""

  def test_get_target_pid_host_cpu(self):
    """Test _get_target_pid resolution for /host:CPU when local_device_id is None."""
    trace_data = {
        "traceEvents": [
            {"pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"pid": 42, "name": "process_name", "args": {"name": "/host:CPU"}},
        ]
    }
    pid = profiler._get_target_pid(  # pylint: disable=protected-access
        trace_data, local_device_id=None
    )
    self.assertEqual(pid, 42)

  def test_get_target_pid_tpu_device(self):
    """Test _get_target_pid resolution for TPU device when local_device_id is provided."""
    trace_data = {
        "traceEvents": [
            {"pid": 10, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"pid": 42, "name": "process_name", "args": {"name": "/host:CPU"}},
        ]
    }
    pid = profiler._get_target_pid(  # pylint: disable=protected-access
        trace_data, local_device_id=0
    )
    self.assertEqual(pid, 10)

  def test_calculate_step_durations_ms_sorts_out_of_order_bounding_events(self):
    """Test that _calculate_step_durations_ms sorts bounding events chronologically."""
    # Bounding events in non-chronological order: step 3, step 1, step 2
    all_events = [
        {"pid": 1, "name": "jit_step3", "ts": 30000, "dur": 5000},
        {"pid": 1, "name": "jit_step1", "ts": 1000, "dur": 5000},
        {"pid": 1, "name": "jit_step2", "ts": 15000, "dur": 5000},
    ]
    candidate_events = [
        {"pid": 1, "ts": 1500, "dur": 1000},  # Step 1: 1.0 ms
        {"pid": 1, "ts": 16000, "dur": 2000},  # Step 2: 2.0 ms
        {"pid": 1, "ts": 31000, "dur": 3000},  # Step 3: 3.0 ms
    ]
    durations = profiler._calculate_step_durations_ms(  # pylint: disable=protected-access
        all_events=all_events,
        candidate_events=candidate_events,
        target_pid=1,
    )
    # Must be sorted chronologically: Step 1 (1.0 ms), Step 2 (2.0 ms), Step 3 (3.0 ms)
    self.assertEqual(durations, [1.0, 2.0, 3.0])


  @mock.patch.object(os, "walk")
  def test_parse_xprof_durations_no_trace(self, mock_walk):
    """Test parse_xprof_durations when no trace file is found."""
    mock_walk.return_value = [("/tmp", [], [])]
    result = profiler.parse_xprof_durations("/tmp")
    self.assertEqual(result, [])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_no_marker_events(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test parse_xprof_durations when there are no marker events in the trace."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    def exists_side_effect(path):
      return path.endswith("trace.json.gz")
    mock_exists.side_effect = exists_side_effect

    # Mock empty trace events
    trace_data = {"traceEvents": []}
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)

    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp")
    self.assertEqual(result, [])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_min_pid_fallback(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test parse_xprof_durations falling back to minimum PID when target pid is missing."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    def exists_side_effect(path):
      return path.endswith("trace.json.gz")
    mock_exists.side_effect = exists_side_effect

    # Mock trace events with marker
    marker = profiler.MARKER
    trace_data = {
        "traceEvents": [
            {
                "pid": 1,
                "dur": 1000,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },
            {
                "pid": 1,
                "dur": 2000,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },
            {
                "pid": 2,
                "dur": 5000,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },  # different pid
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)

    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp")
    self.assertEqual(result, [1.0, 2.0])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_multi_pid_with_idle_device(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test parse_xprof_durations when trace events span multiple PIDs (active and idle devices)."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]

    def exists_side_effect(path):
      return path.endswith("trace.json.gz")

    mock_exists.side_effect = exists_side_effect

    marker = profiler.MARKER
    trace_data = {
        "traceEvents": [
            {
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:TPU:0"},
            },
            {
                "pid": 1,
                "ts": 100,
                "dur": 1000,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },  # active device step 1
            {
                "pid": 2,
                "ts": 100,
                "dur": 10,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },  # idle device step 1
            {
                "pid": 1,
                "ts": 1200,
                "dur": 2000,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },  # active device step 2
            {
                "pid": 2,
                "ts": 1200,
                "dur": 10,
                "args": {"tf_op": f"prefix_{marker}_suffix"},
            },  # idle device step 2
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)

    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp", local_device_id=0)
    # Should take durations for active device PID 1: [1000 us -> 1.0 ms, 2000 us -> 2.0 ms]
    self.assertEqual(result, [1.0, 2.0])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_multiple_target_events_in_jit_enclosure(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test step duration math when multiple target markers occur within a single jit_ enclosure."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    mock_exists.return_value = True

    marker = profiler.MARKER
    trace_data = {
        "traceEvents": [
            {
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:TPU:0"},
            },
            # Bounding enclosure 1: ts=1000, dur=5000 (ends 6000)
            {
                "pid": 1,
                "name": "jit_step1",
                "ts": 1000,
                "dur": 5000,
            },
            # Marker A in step 1: starts at 1500, dur=1000 (ends 2500)
            {
                "pid": 1,
                "ts": 1500,
                "dur": 1000,
                "args": {"tf_op": f"prefix_{marker}_A"},
            },
            # Marker B in step 1: starts at 3000, dur=1500 (ends 4500)
            {
                "pid": 1,
                "ts": 3000,
                "dur": 1500,
                "args": {"tf_op": f"prefix_{marker}_B"},
            },
            # Bounding enclosure 2: ts=10000, dur=4000 (ends 14000)
            {
                "pid": 1,
                "name": "jit_step2",
                "ts": 10000,
                "dur": 4000,
            },
            # Single marker in step 2: starts 11000, dur=2000 (ends 13000)
            {
                "pid": 1,
                "ts": 11000,
                "dur": 2000,
                "args": {"tf_op": f"prefix_{marker}_C"},
            },
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)
    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp", local_device_id=0)
    # Step 1 span: earliest start=1500, latest end=4500 -> (4500-1500)/1000 = 3.0 ms
    # Step 2 span: earliest start=11000, latest end=13000 -> (13000-11000)/1000 = 2.0 ms
    self.assertEqual(result, [3.0, 2.0])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_out_of_order_bounding_enclosures(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test that parse_xprof_durations sorts bounding enclosures chronologically even if emitted out of order."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    mock_exists.return_value = True

    marker = profiler.MARKER
    trace_data = {
        "traceEvents": [
            {
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:TPU:0"},
            },
            # Bounding enclosure 2 is listed first in traceEvents (ts=10000, dur=4000)
            {
                "pid": 1,
                "name": "jit_step2",
                "ts": 10000,
                "dur": 4000,
            },
            # Marker in step 2: starts 11000, dur=2000 (ends 13000 -> 2.0 ms)
            {
                "pid": 1,
                "ts": 11000,
                "dur": 2000,
                "args": {"tf_op": f"prefix_{marker}_step2"},
            },
            # Bounding enclosure 1 is listed second in traceEvents (ts=1000, dur=5000)
            {
                "pid": 1,
                "name": "jit_step1",
                "ts": 1000,
                "dur": 5000,
            },
            # Marker in step 1: starts 1500, dur=1000 (ends 2500 -> 1.0 ms)
            {
                "pid": 1,
                "ts": 1500,
                "dur": 1000,
                "args": {"tf_op": f"prefix_{marker}_step1"},
            },
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)
    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp", local_device_id=0)
    # Output must be ordered chronologically by step time: [1.0 ms (step 1), 2.0 ms (step 2)]
    self.assertEqual(result, [1.0, 2.0])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_host_cpu_dma_fallback_no_jit_enclosure(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test DMA fallback duration extraction on /host:CPU when local_device_id is None."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    mock_exists.return_value = True

    marker = profiler.MARKER
    trace_data = {
        "traceEvents": [
            # Host CPU process metadata
            {
                "pid": 42,
                "name": "process_name",
                "args": {"name": "/host:CPU"},
            },
            # TPU process with noise activity (should be ignored when local_device_id=None)
            {
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:TPU:0"},
            },
            {
                "pid": 1,
                "ts": 100,
                "dur": 500,
                "args": {"tf_op": f"ignored_{marker}_0"},
            },
            # Standalone DMA marker events on /host:CPU (no jit_ enclosures on PID 42)
            {
                "pid": 42,
                "ts": 200,
                "dur": 1200,
                "args": {"tf_op": f"dma_{marker}_1"},
            },
            {
                "pid": 42,
                "ts": 400,
                "args": {
                    "tf_op": f"dma_{marker}_2",
                    "device_duration_ps": 2500000000,  # 2.5 ms in picoseconds
                },
            },
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)
    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp", local_device_id=None)
    # 1200 us = 1.2 ms; 2500000000 ps = 2.5 ms
    self.assertEqual(result, [1.2, 2.5])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_sparsecore_call_done_priority(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test that SparseCore completion markers ('call-done') are prioritized."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    mock_exists.return_value = True

    marker = profiler.MARKER
    trace_data = {
        "traceEvents": [
            {
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:TPU:0"},
            },
            {"pid": 1, "name": "jit_step1", "ts": 100, "dur": 500},
            # Regular marker (should be ignored when call-done exists)
            {
                "pid": 1,
                "ts": 150,
                "dur": 400,
                "args": {"tf_op": f"regular_{marker}"},
            },
            # call-done marker (priority 1 in _extract_candidate_events)
            {
                "pid": 1,
                "name": f"kernel_{marker}_call-done",
                "ts": 200,
                "dur": 150,
            },
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)
    mock_open.return_value.__enter__.return_value = out

    result = profiler.parse_xprof_durations("/tmp", local_device_id=0)
    # Only call-done event duration (150 us = 0.15 ms) should be captured
    self.assertAlmostEqual(result[0], 0.15)
    self.assertLen(result, 1)

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_custom_op_fallback(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test fallback to custom XProf op matching when no markers are present."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    mock_exists.return_value = True

    trace_data = {
        "traceEvents": [
            {
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:TPU:0"},
            },
            {"pid": 1, "name": "jit_step1", "ts": 100, "dur": 1000},
            # Event without standard MARKER string, matched by custom fallback fn
            {
                "pid": 1,
                "ts": 200,
                "dur": 750,
                "args": {"hlo_category": "convolution fusion"},
            },
        ]
    }
    json_str = json.dumps(trace_data)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    out.seek(0)
    mock_open.return_value.__enter__.return_value = out

    def match_fn(e):
      return e.get("args", {}).get("hlo_category", "") == "convolution fusion"

    result = profiler.parse_xprof_durations(
        "/tmp", is_xprof_op_fn=match_fn, local_device_id=0
    )
    self.assertEqual(result, [0.75])


if __name__ == "__main__":
  absltest.main()
