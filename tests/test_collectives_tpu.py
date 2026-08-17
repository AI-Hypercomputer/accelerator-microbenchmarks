"""Unit tests for collectives.py that require TPU devices."""

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
    config = collectives.CollectivesParams(**params)
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


if __name__ == "__main__":
  absltest.main()
