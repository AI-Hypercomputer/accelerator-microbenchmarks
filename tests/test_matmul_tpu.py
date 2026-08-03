"""Unit tests for matlmul.py that require TPU devices."""

import itertools
import os
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import matmul
from accelerator_microbenchmarks.core import registry
import jax
import numpy as np

SUPPORTED_DTYPES = (
    "bfloat16",
    "float8_e4m3fn",
    "int8",
    "float16",
    "float32",
)


class GeneralizedGemmBenchmarkTest(parameterized.TestCase):
  """Unit tests for Generalized GEMM benchmark."""

  def setUp(self):
    super().setUp()
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  @parameterized.named_parameters(
      *[
          {
              "testcase_name": f"{in_dtype}_to_{out_dtype}",
              "in_dtype": in_dtype,
              "out_dtype": out_dtype,
          }
          for in_dtype, out_dtype in itertools.product(
              SUPPORTED_DTYPES, repeat=2
          )
      ]
  )
  @unittest.mock.patch(
      "accelerator_microbenchmarks.core.profiler.upload_xprof_trace"
  )
  def test_real_e2e_matmul_with_xprof(self, mock_upload, in_dtype, out_dtype):
    """Test running a real e2e matmul with real trace collection."""
    mock_upload.return_value = "http://mock_xprof_url"

    # Display detected devices for debugging
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"Detected JAX backend: {backend}")
    print(f"Available JAX devices: {devices}")

    # Assert that a TPU is physically present in the system.
    # This test is meaningless if TPU is not present.
    self.assertEqual(backend, "tpu", "TPU is not present!")
    tpu_devices = jax.devices("tpu")
    print(
        "Verified successfully: test is running on "
        f"real TPU devices: {tpu_devices}"
    )

    # Create a fresh benchmark instance to ensure clean setup and mesh
    # This part captures the profile for debugging use
    undeclared_outputs_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if undeclared_outputs_dir:
      temp_dir = os.path.join(undeclared_outputs_dir, "tpu_tensorboard_profile")
      os.makedirs(temp_dir, exist_ok=True)
      print(
          "Preserving real JAX profiler trace to Undeclared Test Outputs"
          f" directory: {temp_dir}"
      )
    else:
      temp_dir = self.create_tempdir().full_path
      print(f"Saving JAX trace locally to: {temp_dir}")

    params = {
        "m": 128,
        "k": 128,
        "n": 128,
        "in_dtype": in_dtype,
        "out_dtype": out_dtype,
        "warmup_tries": 2,
        "num_runs": 3,
        "xprof_timing": True,
        "xprof_dir": temp_dir,
    }

    config = matmul.GemmParams(**params)
    bm = matmul.GeneralizedGemmBenchmark(config=config, mesh=None)
    bm.setup()
    result = bm.run()
    self.assertIsNotNone(result)
    self.assertEqual(result.metadata.benchmark_name, "GeneralizedGemmBenchmark")
    # Verify that we successfully collected real xprof_avg_ms.
    print("result is", result)
    if "xprof_avg_ms" in result.metrics:
      self.assertGreater(result.metrics["xprof_avg_ms"], 0.0)
    else:
      self.fail("Failed to extract xprof_avg_ms from result")


if __name__ == "__main__":
  absltest.main()
