"""Unit tests for collectives.py that require TPU devices."""

import contextlib
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import collectives
import jax
import numpy as np


class AllReduceTpuTest(parameterized.TestCase):
  """Unit tests for AllReduceBenchmark on TPU hardware."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  @parameterized.named_parameters(
      ("sum", "sum"),
      ("mean", "mean"),
      ("max", "max"),
      ("min", "min"),
  )
  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  def test_all_reduce_ops_tpu(self, reduce_op, mock_upload):
    """Test AllReduceBenchmark on TPU for each reduction operator."""
    backend = jax.default_backend()
    if backend != "tpu":
      self.skipTest("This test requires TPU devices.")

    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "reduce_op": reduce_op,
        "warmup_tries": 1,
        "num_runs": 2,
        "xprof_timing": True,
    }
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(config=config, mesh=self.mock_mesh)
    bm.setup()
    (data,) = bm.generate_inputs()
    out = bm.run_op(data)
    self.assertEqual(out.shape, data.shape)
    self.assertEqual(out.dtype, data.dtype)

    result = bm.run()
    self.assertIsNotNone(result)
    self.assertEqual(result.metadata.benchmark_name, "AllReduceBenchmark")
    self.assertIn("xprof_p50_ms", result.metrics)
    self.assertGreater(result.metrics["xprof_p50_ms"], 0.0)

class AllGatherTpuTest(parameterized.TestCase):
  """Unit tests for AllGatherBenchmark on TPU hardware, including fallback."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  @mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  def test_all_gather_tpu_fallback_without_marker(self, mock_upload):
    """Test AllGatherBenchmark fallback when MARKER is removed."""
    mock_upload.return_value = "http://mock_xprof_url"
    backend = jax.default_backend()
    if backend != "tpu":
      self.skipTest("This test requires TPU devices.")

    undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if undeclared_outputs_dir:
      temp_dir = os.path.join(
          undeclared_outputs_dir, "tpu_tensorboard_profile_all_gather_fallback"
      )
      os.makedirs(temp_dir, exist_ok=True)
    else:
      temp_dir = self.create_tempdir().full_path

    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "warmup_tries": 1,
        "num_runs": 2,
        "xprof_timing": True,
        "xprof_dir": temp_dir,
    }
    config = collectives.CollectivesParams(**params)

    # Use mock.patch on jax.named_scope to run code without the marker scope
    with mock.patch.object(
        jax, "named_scope", side_effect=lambda name: contextlib.nullcontext()
    ):
      bm = collectives.AllGatherBenchmark(config=config, mesh=self.mock_mesh)
      bm.setup()
      (data,) = bm.generate_inputs()
      out = bm.run_op(data)
      self.assertEqual(out.shape, (64 * len(jax.devices()), 8, 128))
      self.assertEqual(out.dtype, data.dtype)
      result = bm.run()
    self.assertIsNotNone(result)

    # 1. Verify that xprof timings were successfully collected via fallback
    self.assertIn("xprof_avg_ms", result.metrics)
    self.assertIsNotNone(
        result.metrics["xprof_avg_ms"],
        "Expected xprof_avg_ms to be populated by fallback when marker is"
        " missing.",
    )
    self.assertGreater(result.metrics["xprof_avg_ms"], 0.0)

if __name__ == "__main__":
  absltest.main()
