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

  @mock.patch.object(os, "walk")
  def test_parse_xprof_results_no_trace(self, mock_walk):
    """Test parse_xprof_results when no trace file is found."""
    mock_walk.return_value = [("/tmp", [], [])]
    metrics = {"existing_metric": 1.0}
    result = profiler.parse_xprof_results("/tmp", "/cns", metrics)
    self.assertEqual(result, metrics)

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_results_no_marker_events(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test parse_xprof_results when there are no marker events."""
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

    metrics = {}
    result = profiler.parse_xprof_results("/tmp", "/cns", metrics)
    self.assertEqual(result, metrics)

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_results_with_metrics(
      self, mock_exists, mock_open, mock_walk
  ):
    """Test parse_xprof_results when valid metrics are found."""
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

    metrics = {}
    result = profiler.parse_xprof_results("/tmp", "/cns", metrics)

    # Should only use events from min_pid (1)
    # durations: 1000 us -> 1 ms, 2000 us -> 2 ms
    # avg: 1.5, p50: 1.5
    self.assertIn("xprof_avg_ms", result)
    self.assertAlmostEqual(result["xprof_avg_ms"], 1.5)
    self.assertAlmostEqual(result["xprof_p50_ms"], 1.5)

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  @mock.patch(
      "google3.perftools.accelerators.xprof.api.python.xprof_analysis_client.XprofAnalysisClient"
  )
  def test_parse_xprof_results_with_upload(
      self, mock_client_class, mock_exists, mock_open, mock_walk
  ):
    """Test parse_xprof_results when xprof upload is successful."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]

    # Return True for trace.json.gz and xplane.pb
    def exists_side_effect(path):
      return path.endswith("trace.json.gz") or path.endswith("xplane.pb")

    mock_exists.side_effect = exists_side_effect

    # Mock file reads
    trace_data = {"traceEvents": []}
    json_str = json.dumps(trace_data)
    trace_out = io.BytesIO()
    with gzip.GzipFile(fileobj=trace_out, mode="w") as f:
      f.write(json_str.encode("utf-8"))
    trace_out.seek(0)

    xplane_out = io.BytesIO(b"")  # Empty bytes for valid empty proto

    def open_side_effect(path, mode="r"):
      _ = mode
      if path.endswith(".gz"):
        return mock.MagicMock(__enter__=mock.MagicMock(return_value=trace_out))
      elif path.endswith("xplane.pb"):
        return mock.MagicMock(
            __enter__=mock.MagicMock(return_value=xplane_out)
        )
      return mock.MagicMock()

    mock_open.side_effect = open_side_effect

    mock_client = mock_client_class.return_value
    mock_client.upload.return_value = "12345"

    metrics = {}
    result = profiler.parse_xprof_results("/tmp", "/cns", metrics)

    self.assertEqual(result.get("xprof_url"), "http://xprof/?session_id=12345")

  @mock.patch.object(os, "walk")
  def test_parse_xprof_durations_no_trace(self, mock_walk):
    """Test parse_xprof_durations when no trace file is found."""
    mock_walk.return_value = [("/tmp", [], [])]
    result = profiler.parse_xprof_durations("/tmp")
    self.assertEqual(result, [])

  @mock.patch.object(os, "walk")
  @mock.patch("builtins.open")
  @mock.patch.object(os.path, "exists")
  def test_parse_xprof_durations_valid(self, mock_exists, mock_open, mock_walk):
    """Test parse_xprof_durations with valid trace events."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]
    def exists_side_effect(path):
      return path.endswith("trace.json.gz")
    mock_exists.side_effect = exists_side_effect

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
  @mock.patch(
      "google3.perftools.accelerators.xprof.api.python.xprof_analysis_client.XprofAnalysisClient"
  )
  def test_upload_xprof_trace_success(
      self, mock_client_class, mock_exists, mock_open, mock_walk
  ):
    """Test upload_xprof_trace when upload is successful."""
    mock_walk.return_value = [("/tmp", [], ["trace.json.gz"])]

    # Return True for xplane.pb and cns_dir to avoid MakeDirs
    def exists_side_effect(path):
      return path.endswith("xplane.pb") or path == "/cns"

    mock_exists.side_effect = exists_side_effect

    xplane_out = io.BytesIO(b"")

    def open_side_effect(path, mode="r"):
      _ = mode
      if path.endswith("xplane.pb"):
        return mock.MagicMock(
            __enter__=mock.MagicMock(return_value=xplane_out)
        )
      return mock.MagicMock()  # Dummy for cns_url_path

    mock_open.side_effect = open_side_effect

    mock_client = mock_client_class.return_value
    mock_client.upload.return_value = "12345"

    result = profiler.upload_xprof_trace("/tmp", "/cns")
    self.assertEqual(result, "http://xprof/?session_id=12345")


if __name__ == "__main__":
  absltest.main()
