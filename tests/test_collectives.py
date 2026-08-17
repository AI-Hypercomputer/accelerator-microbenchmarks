"""Test for collective benchmarks."""

import os

from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
from accelerator_microbenchmarks.benchmarks import collectives
from accelerator_microbenchmarks.core import registry
import jax
import jax.numpy as jnp
import numpy as np


# Force 4 CPU devices for testing collectives
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + " --xla_force_host_platform_device_count=4"
)

# pylint: enable=g-import-not-at-top


# Set CPU backend for fast testing without TPU requirements
jax.config.update("jax_platform_name", "cpu")


class CollectivesBenchmarkTest(parameterized.TestCase):
  """Unit tests for collectives.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing on CPU
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_all_reduce_registered(self):
    """Verify that all_reduce and all_reduce_sum are correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_reduce")
    self.assertEqual(bm_class, collectives.AllReduceBenchmark)
    bm_class_sum = registry.benchmark_registry.get_benchmark("all_reduce_sum")
    self.assertEqual(bm_class_sum, collectives.AllReduceBenchmark)

  def test_all_reduce_invalid_op_raises_error(self):
    """Verify that invalid reduce_op raises ValueError in setup()."""
    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "reduce_op": "invalid_op",
    }
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllReduceBenchmark(config=config, mesh=self.mock_mesh)
    with self.assertRaises(ValueError):
      bm.setup()

  def test_all_reduce_get_run_identifier(self):
    """Verify get_run_identifier returns dim_1024_op_max format."""
    params = {
        "matrix_dim": 1024,
        "reduce_op": "max",
    }
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllReduceBenchmark(config=config, mesh=self.mock_mesh)
    self.assertEqual(bm.get_run_identifier(), "dim_1024_op_max")

  def test_all_gather_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_gather")
    self.assertEqual(bm_class, collectives.AllGatherBenchmark)

  def test_all_to_all_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_to_all")
    self.assertEqual(bm_class, collectives.AllToAllBenchmark)

  def test_reduce_scatter_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("reduce_scatter")
    self.assertEqual(bm_class, collectives.ReduceScatterBenchmark)

  def test_all_reduce_generate_inputs(self):
    """Verify input generation for all_reduce."""
    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
    }
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllReduceBenchmark(config=config, mesh=self.mock_mesh)
    bm.setup()
    (data,) = bm.generate_inputs()
    self.assertEqual(data.shape, (64, 8, 128))
    self.assertEqual(data.dtype, jnp.bfloat16)

  def test_all_gather_generate_inputs(self):
    """Verify input generation for all_gather."""
    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
    }
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllGatherBenchmark(config=config, mesh=self.mock_mesh)
    bm.setup()
    (data,) = bm.generate_inputs()
    self.assertEqual(data.shape, (64, 8, 128))
    self.assertEqual(data.dtype, jnp.bfloat16)

  @parameterized.named_parameters(
      ("sum", "sum", 4.0),
      ("mean", "mean", 1.0),
      ("max", "max", 1.0),
      ("min", "min", 1.0),
  )
  def test_all_reduce_correctness(self, op, factor):
    """Verify numerical correctness for sum, mean, max, min operators."""
    params = {
        "matrix_dim": 2,
        "dtype": "float32",
        "reduce_op": op,
    }
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllReduceBenchmark(config=config, mesh=self.mock_mesh)
    bm.setup()
    (data,) = bm.generate_inputs()
    out = bm.run_op(data)
    expected = factor * np.array(data)
    for shard in out.addressable_shards:
      np.testing.assert_allclose(np.array(shard.data), expected, rtol=1e-5)

  def test_all_gather_with_sharding_strategy(self):
    devices = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=("d_0", "d_1"))
    # Case 1: sharding_strategy = 2x1 (only d_0)
    params_2x1 = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "mesh_shape": "2x2",
        "sharding_strategy": "2x1",
    }
    config_2x1 = collectives.CollectivesParams(**params_2x1)
    bm = collectives.AllGatherBenchmark(config=config_2x1, mesh=mesh)
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x1")
    self.assertEqual(bm._get_sharding_axes(), ("d_0",))

    (data_2x1,) = bm.generate_inputs()
    self.assertEqual(data_2x1.shape, (64, 8, 128))

    out_2x1 = bm.run_op(data_2x1)
    self.assertEqual(out_2x1.shape, (128, 8, 128))

    # Case 2: sharding_strategy = 2x2 (d_0 and d_1)
    params_2x2 = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "mesh_shape": "2x2",
        "sharding_strategy": "2x2",
    }
    config_2x2 = collectives.CollectivesParams(**params_2x2)
    bm = collectives.AllGatherBenchmark(config=config_2x2, mesh=mesh)
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x2")
    self.assertEqual(bm._get_sharding_axes(), ("d_0", "d_1"))

    (data_2x2,) = bm.generate_inputs()
    self.assertEqual(data_2x2.shape, (64, 8, 128))

    out_2x2 = bm.run_op(data_2x2)
    self.assertEqual(out_2x2.shape, (256, 8, 128))

  def test_reduce_scatter_with_sharding_strategy(self):
    devices = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=("d_0", "d_1"))
    # Case 1: sharding_strategy = 2x1 (only d_0)
    params_2x1 = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "mesh_shape": "2x2",
        "sharding_strategy": "2x1",
    }
    config_2x1 = collectives.CollectivesParams(**params_2x1)
    bm = collectives.ReduceScatterBenchmark(config=config_2x1, mesh=mesh)
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x1")
    self.assertEqual(bm._get_sharding_axes(), ("d_0",))

    (data_2x1,) = bm.generate_inputs()
    self.assertEqual(data_2x1.shape, (2, 64, 256))

    out_2x1 = bm.run_op(data_2x1)
    self.assertEqual(out_2x1.shape, (2, 64, 256))

    # Case 2: sharding_strategy = 2x2 (d_0 and d_1)
    params_2x2 = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "mesh_shape": "2x2",
        "sharding_strategy": "2x2",
    }
    config_2x2 = collectives.CollectivesParams(**params_2x2)
    bm = collectives.ReduceScatterBenchmark(config=config_2x2, mesh=mesh)
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x2")
    self.assertEqual(bm._get_sharding_axes(), ("d_0", "d_1"))

    (data_2x2,) = bm.generate_inputs()
    self.assertEqual(data_2x2.shape, (4, 64, 256))

    out_2x2 = bm.run_op(data_2x2)
    self.assertEqual(out_2x2.shape, (4, 64, 256))


if __name__ == "__main__":
  absltest.main()
