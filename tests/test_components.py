"""Unit tests for components.py."""

from absl.testing import absltest
from accelerator_microbenchmarks.benchmarks import components
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import system
import jax
import jax.numpy as jnp
import numpy as np

# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class ComponentsBenchmarkTest(absltest.TestCase):
  """Unit tests for components.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_benchmark_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark(
        "transformer_layer_moe"
    )
    self.assertEqual(bm_class, components.TransformerLayerMoE)

  def test_generate_inputs(self):
    """Verify the shape and type of the generated inputs."""
    params = {
        "model_dim": 256,
        "mslen": 64,
    }
    config = components.TransformerLayerParams(**params)
    self.bm = components.TransformerLayerMoE(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    x, w_attn, b_attn, w_ffn, b_ffn = self.bm.generate_inputs()

    self.assertEqual(x.shape, (1, 64, 256))
    self.assertEqual(w_attn.shape, (256, 256 * 3))
    self.assertEqual(b_attn.shape, (256 * 3, 256))
    self.assertEqual(w_ffn.shape, (256, 256 * 2))
    self.assertEqual(b_ffn.shape, (256 * 2, 256))

    self.assertEqual(x.dtype, jnp.bfloat16)
    self.assertEqual(w_attn.dtype, jnp.bfloat16)
    self.assertEqual(b_attn.dtype, jnp.bfloat16)
    self.assertEqual(w_ffn.dtype, jnp.bfloat16)
    self.assertEqual(b_ffn.dtype, jnp.bfloat16)

  def test_run_op(self):
    """Verify that running the op returns the expected shape."""
    params = {
        "model_dim": 256,
        "mslen": 64,
    }
    config = components.TransformerLayerParams(**params)
    self.bm = components.TransformerLayerMoE(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.bm.setup()
    inputs = self.bm.generate_inputs()
    out = self.bm.run_op(*inputs)

    self.assertEqual(out.shape, (1, 64, 256))
    self.assertEqual(out.dtype, jnp.bfloat16)

  def test_get_total_bytes(self):
    """Verify the byte calculation."""
    params = {
        "model_dim": 256,
        "mslen": 64,
    }
    config = components.TransformerLayerParams(**params)
    self.bm = components.TransformerLayerMoE(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # model_dim = 256
    # seq_len = 64
    # itemsize = 2 (bfloat16)
    # x_size = 64 * 256 * 2 = 32768
    # w_size = (256 * 256 * 10) * 2 = 1310720
    # Total = 10 * 32768 + 1310720 = 327680 + 1310720 = 1638400
    expected_bytes = 1638400.0
    self.assertAlmostEqual(self.bm.get_total_bytes(), expected_bytes)

  def test_get_arithmetic_intensity(self):
    """Verify the intensity calculation."""
    params = {
        "model_dim": 256,
        "mslen": 64,
    }
    config = components.TransformerLayerParams(**params)
    self.bm = components.TransformerLayerMoE(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # flops = 24 * 64 * (256^2) = 1536 * 65536 = 100663296
    # bytes = 1638400
    # intensity = 100663296 / 1638400 = 61.44
    expected_intensity = 61.44
    self.assertAlmostEqual(
        self.bm.get_arithmetic_intensity(), expected_intensity
    )

  def test_calculate_metrics(self):
    """Verify that metrics are correctly calculated."""
    params = {
        "model_dim": 256,
        "mslen": 64,
    }
    config = components.TransformerLayerParams(**params)
    self.bm = components.TransformerLayerMoE(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    # avg_ms = 10.0ms -> avg_latency_s = 0.01s
    # flops = 100663296 (from above)
    # tflops_per_sec = (100663296 / 0.01) / 1e12 = 10066329600 / 1e12 = 0.010066
    times_ms = [10.0, 10.0, 10.0]
    metrics = self.bm.calculate_metrics(times_ms)

    self.assertAlmostEqual(metrics["avg_ms"], 10.0)
    self.assertAlmostEqual(metrics["intensity"], 61.44)
    self.assertAlmostEqual(metrics["tflops_per_sec"], 0.0100663296)


if __name__ == "__main__":
  absltest.main()
