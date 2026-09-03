"""Test for collective benchmarks."""

import dataclasses
import os

from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-import-not-at-top
from accelerator_microbenchmarks.benchmarks import collectives
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
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

_COLLECTIVES_IGNORED_KEYS: frozenset[str] = frozenset({
    "replica_group_type",
    "replica_group_rank",
})


class CollectivesBenchmarkTest(parameterized.TestCase):
  """Unit tests for collectives.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing on CPU
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )

  def test_all_reduce_registered(self):
    """Verify that all_reduce is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_reduce")
    self.assertEqual(bm_class, collectives.AllReduceBenchmark)
    self.assertEqual(bm_class.Config, collectives.AllReduceParams)

  def test_all_reduce_invalid_op_raises_error(self):
    """Verify that invalid reduce_op raises ValueError in setup()."""
    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "reduce_op": "invalid_op",
    }
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    with self.assertRaises(ValueError):
      bm.setup()

  def test_all_reduce_get_run_identifier(self):
    """Verify get_run_identifier returns dim_1024_op_max format."""
    params = {
        "matrix_dim": 1024,
        "reduce_op": "max",
    }
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.assertEqual(bm.get_run_identifier(), "dim_1024_op_max")

  def test_all_gather_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_gather")
    self.assertEqual(bm_class, collectives.AllGatherBenchmark)

  def test_all_gather_match_xprof_op_fallback(self):
    """Test that match_xprof_op_fallback correctly identifies async-done events."""
    config = collectives.CollectivesParams(matrix_dim=64)
    bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    self.assertTrue(
        bm.match_xprof_op_fallback({
            "args": {
                "hlo_category": "async-done",
                "offload_type": "OFFLOAD_COLLECTIVE",
            }
        })
    )

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
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
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
    bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
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
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
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
    bm = collectives.AllGatherBenchmark(
        config=config_2x1, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
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
    bm = collectives.AllGatherBenchmark(
        config=config_2x2, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
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
    bm = collectives.ReduceScatterBenchmark(
        config=config_2x1, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
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
    bm = collectives.ReduceScatterBenchmark(
        config=config_2x2, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x2")
    self.assertEqual(bm._get_sharding_axes(), ("d_0", "d_1"))

    (data_2x2,) = bm.generate_inputs()
    self.assertEqual(data_2x2.shape, (4, 64, 256))

    out_2x2 = bm.run_op(data_2x2)
    self.assertEqual(out_2x2.shape, (4, 64, 256))

  def test_transfer_metrics_calculation(self):
    devices = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=("d_0", "d_1"))
    config = collectives.CollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy="2x2",
    )
    # AllGather
    ag_bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
    ag_bm.setup()
    ag_metrics = ag_bm.calculate_metrics([1.0])  # 1.0 ms latency
    # local_size = 1024 * 8 * 128 * 4 = 4194304 bytes
    # non-parallel: participating_ranks = 4 - 2 = 2
    # data_transferred = 4194304 * 2 = 8388608 bytes
    # avg_latency = 0.001 s -> bandwidth = 8.388608 GB/s
    self.assertAlmostEqual(ag_metrics["bandwidth_gb_s"], 8.388608, places=4)

    # AllReduce
    ar_config = collectives.AllReduceParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy="2x2",
        reduce_op="sum",
    )
    ar_bm = collectives.AllReduceBenchmark(
        config=ar_config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
    ar_bm.setup()
    ar_metrics = ar_bm.calculate_metrics([1.0])
    # local_size = 4194304 bytes
    # data_transferred = 2 * 4194304 * (2 / 4) = 4194304 bytes
    # avg_latency = 0.001 s -> bandwidth = 4.194304 GB/s
    self.assertAlmostEqual(ar_metrics["bandwidth_gb_s"], 4.194304, places=4)

  def test_replica_groups_hlo_parsing(self):
    devices = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=("d_0", "d_1"))

    # Parallel replica groups (strided)
    dump_dir_p = self.create_tempdir().full_path
    with open(os.path.join(dump_dir_p, "after_optimizations.txt"), "w") as f:
      f.write("HloModule ... replica_groups={{0,2,4,6},{1,3,5,7}}")

    config_parallel = collectives.CollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy="2x2",
        xla_dump_dir=dump_dir_p,
    )
    ag_parallel = collectives.AllGatherBenchmark(
        config=config_parallel, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
    ag_parallel.setup()
    metrics_p = ag_parallel.calculate_metrics([1.0])
    self.assertEqual(metrics_p["replica_group_type"], "parallel")
    self.assertEqual(metrics_p["replica_group_rank"], 4)

    # Non-parallel replica groups (contiguous)
    dump_dir_np = self.create_tempdir().full_path
    with open(os.path.join(dump_dir_np, "after_optimizations.txt"), "w") as f:
      f.write("HloModule ... replica_groups={{0,1,2,3},{4,5,6,7}}")

    config_non_parallel = collectives.CollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy="2x2",
        xla_dump_dir=dump_dir_np,
    )
    ag_non_parallel = collectives.AllGatherBenchmark(
        config=config_non_parallel, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
    ag_non_parallel.setup()
    metrics_np = ag_non_parallel.calculate_metrics([1.0])
    self.assertEqual(metrics_np["replica_group_type"], "non-parallel")
    self.assertEqual(metrics_np["replica_group_rank"], 4)

  def test_calculate_metrics_exception_fallback(self):
    devices = np.array(jax.devices()).reshape((2, 2))
    mesh = jax.sharding.Mesh(devices, axis_names=("d_0", "d_1"))

    config = collectives.CollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy="2x2",
    )
    ag_bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=mesh
    )
    ag_bm.setup()

    def mock_extract_raises():
      raise ValueError("Mocked error")

    ag_bm._extract_first_replica_group_from_hlo_dump = mock_extract_raises

    metrics = ag_bm.calculate_metrics([1.0])
    self.assertEqual(metrics["replica_group_type"], "non-parallel")
    self.assertEqual(metrics["replica_group_rank"], 4)

  def test_format_benchmark_table(self):
    """Tests formatting of collective benchmark tables."""
    res_base = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="AllReduceBenchmark",
            test_name="AllReduceBenchmark_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "matrix_dim": 4096,
                "mesh_shape": "2x2x2",
                "reduce_op": "sum",
                "sharding_strategy": "2x2x1",
                "dtype": "bfloat16",
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "shard_size_mib": 32.0,
            "p50_ms": 0.05201,
            "bandwidth_gb_s": 350.123,
            "xprof_p50_ms": 0.04812,
        },
        raw_times_ms=[1.0],
    )
    expected_cols = [
        "dtype",
        "reduce_op",
        "mesh_shape",
        "sharding_strategy",
        "matrix_dim",
        "shard_size_mib",
        "bandwidth_gb_s",
        "p50_ms",
        "xprof_p50_ms",
    ]
    schema_cols = [
        col for col, _ in collectives.AllReduceBenchmark.REPORT_SCHEMA
    ]
    self.assertEqual(schema_cols, expected_cols)

    df_ar = report.results_to_dataframe([res_base])
    table_ar = report.format_benchmark_table(
        df_ar,
        schema=collectives.AllReduceBenchmark.REPORT_SCHEMA,
        title="AllReduceBenchmark",
    )
    self.assertIn("Benchmark Results (AllReduceBenchmark)", table_ar)
    for col in expected_cols:
      self.assertIn(col, table_ar)
    self.assertIn("bfloat16", table_ar)
    self.assertIn("sum", table_ar)
    self.assertIn("2x2x2", table_ar)
    self.assertIn("2x2x1", table_ar)
    self.assertIn("4096", table_ar)
    self.assertIn("32.00", table_ar)
    self.assertIn("350.12", table_ar)
    self.assertIn("0.0520", table_ar)
    self.assertIn("0.0481", table_ar)

    res_a2a = base.BenchmarkResult(
        metadata=base.BenchmarkMetadata(
            benchmark_name="AllToAllBenchmark",
            test_name="AllToAllBenchmark_test",
            start_time="2026-08-18T10:00:00",
            end_time="2026-08-18T10:01:00",
            params={
                "matrix_dim": 4096,
                "mesh_shape": "2x2x2",
                "sharding_strategy": "2x2x1",
                "dtype": "bfloat16",
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "local_size_mib": 64.0,
            "p50_ms": 0.05201,
            "bandwidth_gb_s": 350.123,
            "xprof_p50_ms": 0.04812,
        },
        raw_times_ms=[1.0],
    )
    expected_a2a_cols = [
        "dtype",
        "mesh_shape",
        "sharding_strategy",
        "matrix_dim",
        "local_size_mib",
        "bandwidth_gb_s",
        "p50_ms",
        "xprof_p50_ms",
    ]
    schema_a2a_cols = [
        col for col, _ in collectives.AllToAllBenchmark.REPORT_SCHEMA
    ]
    self.assertEqual(schema_a2a_cols, expected_a2a_cols)

    df_a2a = report.results_to_dataframe([res_a2a])
    table_a2a = report.format_benchmark_table(
        df_a2a,
        schema=collectives.AllToAllBenchmark.REPORT_SCHEMA,
        title="AllToAllBenchmark",
    )
    self.assertIn("Benchmark Results (AllToAllBenchmark)", table_a2a)
    for col in expected_a2a_cols:
      self.assertIn(col, table_a2a)
    self.assertIn("bfloat16", table_a2a)
    self.assertIn("2x2x2", table_a2a)
    self.assertIn("2x2x1", table_a2a)
    self.assertIn("4096", table_a2a)
    self.assertIn("64.00", table_a2a)
    self.assertIn("350.12", table_a2a)
    self.assertIn("0.0520", table_a2a)
    self.assertIn("0.0481", table_a2a)

  def test_all_reduce_schema_coverage(self):
    """Verify AllReduceBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  def test_all_gather_schema_coverage(self):
    """Verify AllGatherBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  def test_all_to_all_schema_coverage(self):
    """Verify AllToAllBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.CollectivesParams(**params)
    bm = collectives.AllToAllBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  def test_reduce_scatter_schema_coverage(self):
    """Verify ReduceScatterBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.CollectivesParams(**params)
    bm = collectives.ReduceScatterBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  def test_collectives_params_hierarchy(self):
    """Verify inheritance and field segregation between CollectivesParams and AllReduceParams."""
    self.assertTrue(
        issubclass(collectives.AllReduceParams, collectives.CollectivesParams)
    )
    self.assertTrue(
        issubclass(collectives.CollectivesParams, base.BaseBenchmarkParams)
    )

    all_reduce_fields = {
        f.name for f in dataclasses.fields(collectives.AllReduceParams)
    }
    self.assertIn("reduce_op", all_reduce_fields)


if __name__ == "__main__":
  absltest.main()
