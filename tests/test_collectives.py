"""Test for collective benchmarks."""

import argparse
import dataclasses
import io
import itertools
import math
import os
from typing import Any, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from accelerator_microbenchmarks import cli
from accelerator_microbenchmarks.benchmarks import collectives
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.tests import test_report_utils
import jax
import jax.numpy as jnp
import numpy as np


def setUpModule():  # pylint: disable=invalid-name
  # Force 4 CPU devices for testing collectives
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=4"
  )
  # Set CPU backend for fast testing without TPU requirements
  jax.config.update("jax_platform_name", "cpu")


_ALL_DIMS_GE_4 = system.WraparoundRule.ALL_DIMS_GE_4
_PER_DIM_GE_16 = system.WraparoundRule.PER_DIM_GE_16
_NO_WRAP = system.WraparoundRule.NONE
_LINK_BW = 100.0


def _box_coords(dims: Sequence[int]) -> list[tuple[int, ...]]:
  """Returns the chip coordinates of a solid box starting at the origin."""
  return list(itertools.product(*(range(d) for d in dims)))


def _peak(
    rank: int,
    k: int,
    group_dims: Sequence[int],
    topology_dims: Sequence[int],
    rule: system.WraparoundRule,
) -> float:
  return collectives.all_to_all_peak_bandwidth_per_chip_gb_s(
      replica_group_rank=rank,
      devices_per_chip_in_replica_group=k,
      replica_group_chip_coords=_box_coords(group_dims),
      topology_dims=topology_dims,
      link_bw_per_direction_gb_s=_LINK_BW,
      wraparound_rule=rule,
  )


_COLLECTIVES_IGNORED_KEYS: frozenset[str] = frozenset({
    "data_transferred_bytes",
    "replica_group_type",
    "replica_group_rank",
})

# Default and metadata of every collective param field, keyed by field name.
# Compared by name only, so it does not pin the dataclass field order.
_HEAD_FIELD_SPEC: dict[str, tuple[object, dict[str, object]]] = {
    "warmup_tries": (
        10,
        {"min": 0, "help": "Number of warmup iterations before measurement."},
    ),
    "num_runs": (
        10,
        {"min": 0, "help": "Number of measurement iterations."},
    ),
    "min_duration_s": (
        0.0,
        {
            "min": 0.0,
            "help": (
                "Minimum measurement duration in seconds. Also runs warmup for"
                " at least min_duration_s / 5 (capped at 1.0s)."
            ),
        },
    ),
    "dtype": (
        "bfloat16",
        {"help": "Data type for tensor operations (e.g. bfloat16, float32)."},
    ),
    "matrix_dim": (
        1024,
        {
            "min": 1,
            "help": (
                "Dimension size for sharded collective matrices. Actual matrix"
                " size will be (matrix_dim, 8, 128)."
            ),
        },
    ),
    "seed": (
        0,
        {"min": 0, "help": "Random seed for tensor initialization."},
    ),
    "mesh_shape": (
        None,
        {"help": "Logical TPU mesh shape string (e.g. '2x4x4', '2x2x1')."},
    ),
    "device_placement": (
        collectives.DevicePlacement.MESH_UTILS,
        {
            "help": (
                "Mesh device placement: 'mesh_utils' (topology-aware) or"
                " 'id_sorted' (sorted by device id)."
            )
        },
    ),
    "sharding_strategy": (
        None,
        {
            "help": (
                "Device axis sharding strategy string (e.g. '2x2x2', '2x2x1')."
            )
        },
    ),
    "reduce_op": (
        collectives.ReduceOp.SUM,
        {"help": "Reduction operation."},
    ),
}


@dataclasses.dataclass
class _MockTpuDevice:
  id: int
  core_on_chip: int
  coords: tuple[int, ...] | None = None
  slice_index: int = 0


def _mock_tpu7x_devices(
    topology: tuple[int, int, int],
    cores: tuple[int, ...] = (0, 1),
    slice_index: int = 0,
) -> list[_MockTpuDevice]:
  """Returns mock TPU7x devices (2 cores per chip) for a chip-level topology."""
  chip_coords = itertools.product(*(range(d) for d in topology))
  return [
      _MockTpuDevice(
          id=2 * chip + core,
          core_on_chip=core,
          coords=coords,
          slice_index=slice_index,
      )
      for chip, coords in enumerate(chip_coords)
      for core in cores
  ]


class CollectivesBenchmarkTest(parameterized.TestCase):
  """Unit tests for collectives.py."""

  def setUp(self):
    super().setUp()
    # Create a dummy mesh for testing on CPU
    self.mock_mesh = jax.sharding.Mesh(
        np.array(jax.devices()), axis_names=("device",)
    )
    # CPU devices lack core_on_chip, so tag each first replica group device
    # with id % 2 (TPU7x device ids are 2 * chip + core).
    bm_cls = collectives.BaseCollectiveBenchmark
    get_devices = bm_cls._get_first_replica_group_devices  # pylint: disable=protected-access
    self.enter_context(
        mock.patch.object(
            bm_cls,
            "_get_first_replica_group_devices",
            autospec=True,
            side_effect=lambda bm: [
                _MockTpuDevice(id=d.id, core_on_chip=d.id % 2)
                for d in get_devices(bm)
            ],
        )
    )

  def test_all_reduce_registered(self):
    """Verify that all_reduce is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_reduce")
    self.assertEqual(bm_class, collectives.AllReduceBenchmark)
    self.assertEqual(bm_class.Config, collectives.AllReduceParams)

  def test_reduction_collectives_complex_dtype_raises_error(self):
    """Verify all-reduce/reduce-scatter reject complex64, unlike the others."""
    collectives.IciCollectivesParams(dtype="complex64")
    collectives.AllReduceParams(dtype="int8")
    collectives.ReduceScatterParams(dtype="int8")
    for params_cls in (
        collectives.AllReduceParams,
        collectives.DcnAllReduceParams,
        collectives.ReduceScatterParams,
    ):
      with self.assertRaisesRegex(ValueError, "Unsupported dtype: 'complex64'"):
        params_cls(dtype="complex64")

  def test_all_reduce_invalid_op_raises_error(self):
    """Verify that invalid reduce_op raises ValueError at initialization."""
    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "reduce_op": "invalid_op",
    }
    with self.assertRaisesRegex(ValueError, "Invalid reduce_op 'invalid_op'"):
      collectives.AllReduceParams(**params)

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

  def test_dcn_all_reduce_registered(self):
    """Verify that dcn_all_reduce is correctly registered with DcnAllReduceParams."""
    bm_class = registry.benchmark_registry.get_benchmark("dcn_all_reduce")
    self.assertEqual(bm_class, collectives.DcnAllReduceBenchmark)
    self.assertEqual(bm_class.Config, collectives.DcnAllReduceParams)

  def test_all_gather_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_gather")
    self.assertEqual(bm_class, collectives.AllGatherBenchmark)
    self.assertEqual(bm_class.Config, collectives.IciCollectivesParams)

  def test_all_to_all_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("all_to_all")
    self.assertEqual(bm_class, collectives.AllToAllBenchmark)
    self.assertEqual(bm_class.Config, collectives.IciCollectivesParams)

  def test_reduce_scatter_registered(self):
    """Verify that the benchmark is correctly registered."""
    bm_class = registry.benchmark_registry.get_benchmark("reduce_scatter")
    self.assertEqual(bm_class, collectives.ReduceScatterBenchmark)
    self.assertEqual(bm_class.Config, collectives.ReduceScatterParams)

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
    config = collectives.IciCollectivesParams(**params)
    bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC, mesh=self.mock_mesh
    )
    bm.setup()
    (data,) = bm.generate_inputs()
    self.assertEqual(data.shape, (64, 8, 128))
    self.assertEqual(data.dtype, jnp.bfloat16)

  def test_all_to_all_generate_inputs(self):
    """Verify input generation for all_to_all."""
    params = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
    }
    config = collectives.IciCollectivesParams(**params)
    bm = collectives.AllToAllBenchmark(
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
    # Case 1: sharding_strategy = 2x1 (only d_0)
    params_2x1 = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "mesh_shape": "2x2",
        "sharding_strategy": "2x1",
    }
    config_2x1 = collectives.IciCollectivesParams(**params_2x1)
    bm = collectives.AllGatherBenchmark(
        config=config_2x1, hardware_spec=system.TPU7X_HARDWARE_SPEC
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
    config_2x2 = collectives.IciCollectivesParams(**params_2x2)
    bm = collectives.AllGatherBenchmark(
        config=config_2x2, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x2")
    self.assertEqual(bm._get_sharding_axes(), ("d_0", "d_1"))

    (data_2x2,) = bm.generate_inputs()
    self.assertEqual(data_2x2.shape, (64, 8, 128))

    out_2x2 = bm.run_op(data_2x2)
    self.assertEqual(out_2x2.shape, (256, 8, 128))

  def test_reduce_scatter_with_sharding_strategy(self):
    # Case 1: sharding_strategy = 2x1 (only d_0)
    params_2x1 = {
        "matrix_dim": 64,
        "dtype": "bfloat16",
        "mesh_shape": "2x2",
        "sharding_strategy": "2x1",
    }
    config_2x1 = collectives.IciCollectivesParams(**params_2x1)
    bm = collectives.ReduceScatterBenchmark(
        config=config_2x1, hardware_spec=system.TPU7X_HARDWARE_SPEC
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
    config_2x2 = collectives.IciCollectivesParams(**params_2x2)
    bm = collectives.ReduceScatterBenchmark(
        config=config_2x2, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    bm.setup()
    self.assertEqual(bm.sharding_strategy, "2x2")
    self.assertEqual(bm._get_sharding_axes(), ("d_0", "d_1"))

    (data_2x2,) = bm.generate_inputs()
    self.assertEqual(data_2x2.shape, (4, 64, 256))

    out_2x2 = bm.run_op(data_2x2)
    self.assertEqual(out_2x2.shape, (4, 64, 256))

  def test_all_to_all_with_sharding_strategy(self):
    # Case 1: 2D mesh (V6E_HARDWARE_SPEC)
    for sharding_strategy, expected_axes in [
        ("2x1", ("d_0",)),
        ("2x2", ("d_0", "d_1")),
    ]:
      config_2d = collectives.IciCollectivesParams(
          matrix_dim=64,
          dtype="bfloat16",
          mesh_shape="2x2",
          sharding_strategy=sharding_strategy,
      )
      bm_2d = collectives.AllToAllBenchmark(
          config=config_2d,
          hardware_spec=system.V6E_HARDWARE_SPEC,
      )
      bm_2d.setup()
      self.assertEqual(bm_2d.sharding_strategy, sharding_strategy)
      self.assertEqual(bm_2d._get_sharding_axes(), expected_axes)
      (data_2d,) = bm_2d.generate_inputs()
      self.assertEqual(data_2d.shape, (64, 8, 128))
      out_2d = bm_2d.run_op(data_2d)
      self.assertEqual(out_2d.shape, (64, 8, 128))

    # Case 2: 3D mesh (TPU7X_HARDWARE_SPEC)
    for sharding_strategy, expected_axes in [
        ("2x1x1", ("d_0",)),
        ("2x2x1", ("d_0", "d_1")),
    ]:
      config_3d = collectives.IciCollectivesParams(
          matrix_dim=64,
          dtype="bfloat16",
          mesh_shape="2x2x1",
          sharding_strategy=sharding_strategy,
      )
      bm_3d = collectives.AllToAllBenchmark(
          config=config_3d,
          hardware_spec=system.TPU7X_HARDWARE_SPEC,
      )
      bm_3d.setup()
      self.assertEqual(bm_3d.sharding_strategy, sharding_strategy)
      self.assertEqual(bm_3d._get_sharding_axes(), expected_axes)
      (data_3d,) = bm_3d.generate_inputs()
      self.assertEqual(data_3d.shape, (64, 8, 128))
      out_3d = bm_3d.run_op(data_3d)
      self.assertEqual(out_3d.shape, (64, 8, 128))

  @parameterized.named_parameters(
      (
          "all_gather",
          collectives.AllGatherBenchmark,
          collectives.IciCollectivesParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2x1",
              sharding_strategy="2x2x1",
          ),
          "shard_size_mib",
      ),
      (
          "all_reduce",
          collectives.AllReduceBenchmark,
          collectives.AllReduceParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2x1",
              sharding_strategy="2x2x1",
              reduce_op="sum",
          ),
          "shard_size_mib",
      ),
      (
          "all_to_all",
          collectives.AllToAllBenchmark,
          collectives.IciCollectivesParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2x1",
              sharding_strategy="2x2x1",
          ),
          "shard_size_mib",
      ),
  )
  def test_input_shape_and_transfer_metrics_invariant(
      self, benchmark_cls, config, size_metric_key
  ):
    """Guardrail ensuring generated input tensor bytes strictly equal local_size_bytes in transfer metrics."""
    bm = benchmark_cls(config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC)
    bm.setup()
    (data,) = bm.generate_inputs()
    metrics = bm.calculate_metrics([1.0])

    actual_input_bytes = data.size * data.dtype.itemsize
    metrics_local_bytes = int(round(metrics[size_metric_key] * 1024 * 1024))
    self.assertEqual(
        actual_input_bytes,
        metrics_local_bytes,
        f"For {benchmark_cls.__name__}, generated input tensor bytes"
        f" ({actual_input_bytes}) must strictly equal {size_metric_key} in"
        f" transfer metrics ({metrics_local_bytes}) to prevent shape/metric"
        " disconnects.",
    )

  @parameterized.named_parameters(
      (
          "all_gather_tpu7x",
          collectives.AllGatherBenchmark,
          collectives.IciCollectivesParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2",
              sharding_strategy="2x2",
          ),
          system.TPU7X_HARDWARE_SPEC,
          8.388608,
      ),
      (
          "all_gather_v6e",
          collectives.AllGatherBenchmark,
          collectives.IciCollectivesParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2",
              sharding_strategy="2x2",
          ),
          system.V6E_HARDWARE_SPEC,
          12.582912,
      ),
      (
          "all_reduce_tpu7x",
          collectives.AllReduceBenchmark,
          collectives.AllReduceParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2",
              sharding_strategy="2x2",
              reduce_op="sum",
          ),
          system.TPU7X_HARDWARE_SPEC,
          4.194304,
      ),
      (
          "all_reduce_v6e",
          collectives.AllReduceBenchmark,
          collectives.AllReduceParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2",
              sharding_strategy="2x2",
              reduce_op="sum",
          ),
          system.V6E_HARDWARE_SPEC,
          6.291456,
      ),
      (
          "all_to_all_tpu7x",
          collectives.AllToAllBenchmark,
          collectives.IciCollectivesParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2",
              sharding_strategy="2x2",
          ),
          system.TPU7X_HARDWARE_SPEC,
          4.194304,
      ),
      (
          "all_to_all_v6e",
          collectives.AllToAllBenchmark,
          collectives.IciCollectivesParams(
              matrix_dim=1024,
              dtype="float32",
              mesh_shape="2x2",
              sharding_strategy="2x2",
          ),
          system.V6E_HARDWARE_SPEC,
          3.145728,
      ),
  )
  def test_transfer_metrics_calculation(
      self, benchmark_cls, config, hardware_spec, expected_bw
  ):
    bm = benchmark_cls(config=config, hardware_spec=hardware_spec)
    bm.setup()
    metrics = bm.calculate_metrics([1.0])
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_chip_gb_s"], expected_bw, places=4
    )
    self.assertNotIn("wall_clock_bandwidth_per_device_gb_s", metrics)
    self.assertNotIn("sharding_size", metrics)
    derived = bm.derive_chip_metrics(dict(metrics))
    self.assertNotIn("wall_clock_bandwidth_per_device_gb_s", derived)

  @parameterized.named_parameters(
      ("parallel", "2x1", "parallel", 2, 8.388608),
      ("non_parallel", "2x2", "non-parallel", 4, 8.388608),
  )
  def test_mesh_based_parallel_vs_non_parallel_metrics(
      self, sharding_strategy, expected_group_type, expected_rank, expected_bw
  ):
    """Verify parallel vs non-parallel detection directly from mesh device layout without HLO dump files."""
    config = collectives.IciCollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy=sharding_strategy,
    )
    ag_bm = collectives.AllGatherBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    ag_bm.setup()
    metrics = ag_bm.calculate_metrics([1.0])
    self.assertEqual(metrics["replica_group_type"], expected_group_type)
    self.assertEqual(metrics["replica_group_rank"], expected_rank)
    self.assertNotIn("sharding_size", metrics)
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_chip_gb_s"], expected_bw, places=4
    )

  @parameterized.named_parameters(
      (
          "mesh_parallel_2_chips",
          "2x1",
          None,
          "parallel",
          2,
          4194304.0,
          4.194304,
      ),
      (
          "mesh_non_parallel_2_chips",
          "2x2",
          None,
          "non-parallel",
          4,
          4194304.0,
          4.194304,
      ),
      (
          "parallel_8_chips",
          "2x2",
          [0, 2, 4, 6, 8, 10, 12, 14],
          "parallel",
          8,
          7340032.0,
          7.340032,
      ),
      (
          "non_parallel_8_chips",
          "2x2",
          list(range(16)),
          "non-parallel",
          16,
          7340032.0,
          7.340032,
      ),
  )
  def test_all_to_all_parallel_vs_non_parallel_transfer_metrics(
      self,
      sharding_strategy,
      group_ids,
      expected_group_type,
      expected_rank,
      expected_bytes,
      expected_bw,
  ):
    """Verify AllToAllBenchmark counts all devices_per_chip cores in both parallel and non-parallel modes."""
    config = collectives.IciCollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy=sharding_strategy,
    )
    a2a_bm = collectives.AllToAllBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    a2a_bm.setup()
    self.assertEqual(a2a_bm.hardware_spec.devices_per_chip, 2)
    if group_ids is not None:
      # Larger TPU7x replica group than the 4 CPU devices can form.
      a2a_bm._get_first_replica_group_devices = lambda: [  # pylint: disable=protected-access
          _MockTpuDevice(id=i, core_on_chip=i % 2) for i in group_ids
      ]

    metadata = a2a_bm.get_workload_metadata()
    self.assertEqual(metadata["replica_group_type"], expected_group_type)
    self.assertEqual(metadata["replica_group_rank"], expected_rank)
    self.assertAlmostEqual(
        metadata["data_transferred_bytes"], expected_bytes, places=1
    )

    metrics = a2a_bm.calculate_metrics([1.0])
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_chip_gb_s"], expected_bw, places=4
    )

  def _make_all_to_all_ici_benchmark(
      self,
      replica_group: list[_MockTpuDevice],
      topology_devices: list[Any],
      hardware_spec: system.HardwareSpec = system.TPU7X_HARDWARE_SPEC,
  ) -> collectives.AllToAllBenchmark:
    """Returns an AllToAllBenchmark with mocked replica group and topology."""
    a2a_bm = collectives.AllToAllBenchmark(
        config=collectives.CollectivesParams(matrix_dim=64),
        hardware_spec=hardware_spec,
        mesh=self.mock_mesh,
    )
    a2a_bm._get_first_replica_group_devices = lambda: replica_group  # pylint: disable=protected-access
    a2a_bm._get_topology_devices = lambda: topology_devices  # pylint: disable=protected-access
    return a2a_bm

  def test_all_to_all_roofline_mode_is_network_ici(self):
    self.assertEqual(
        collectives.AllToAllBenchmark.roofline_mode,
        constants.RooflineMode.NETWORK_ICI,
    )

  @parameterized.named_parameters(
      ("non_parallel_2x2x1", (2, 2, 1), (2, 2, 1), (0, 1), 150.0),
      ("parallel_2x2x1", (2, 2, 1), (2, 2, 1), (0,), 150.0),
      ("non_parallel_4x4x4", (4, 4, 4), (4, 4, 4), (0, 1), 196.875),
      # Replica group spans a subset of the ICI topology (a 4x4 plane).
      ("parallel_4x4_in_4x4x4", (4, 4, 1), (4, 4, 4), (0,), 187.5),
  )
  def test_all_to_all_peak_ici_bandwidth(
      self, group_topology, topology, group_cores, expected_peak
  ):
    """Verify the ICI peak is modeled from mocked TPU7x devices."""
    a2a_bm = self._make_all_to_all_ici_benchmark(
        replica_group=_mock_tpu7x_devices(group_topology, cores=group_cores),
        topology_devices=_mock_tpu7x_devices(topology),
    )
    with self.assertNoLogs(level="WARNING"):
      peak = a2a_bm.get_peak_ici_bandwidth_per_chip()
    self.assertAlmostEqual(peak, expected_peak)

  def test_all_to_all_apply_roofline_analysis_emits_ici_efficiency(self):
    a2a_bm = self._make_all_to_all_ici_benchmark(
        replica_group=_mock_tpu7x_devices((2, 2, 1)),
        topology_devices=_mock_tpu7x_devices((2, 2, 1)),
    )
    metrics = a2a_bm.apply_roofline_analysis(
        {"wall_clock_bandwidth_per_chip_gb_s": 75.0}
    )
    self.assertAlmostEqual(metrics["peak_ici_bandwidth_per_chip_gb_s"], 150.0)
    self.assertAlmostEqual(
        metrics["wall_clock_ici_roofline_efficiency_pct"], 50.0
    )

  def test_all_to_all_ici_skipped_without_device_coords(self):
    """CPU devices have no physical coordinates, so no ICI peak is modeled."""
    a2a_bm = self._make_all_to_all_ici_benchmark(
        replica_group=[_MockTpuDevice(id=0, core_on_chip=0)],
        topology_devices=list(jax.devices()),
    )
    with self.assertLogs(level="WARNING") as logs:
      self.assertIsNone(a2a_bm.get_peak_ici_bandwidth_per_chip())
    self.assertIn("no physical chip coordinates", logs.output[0])
    metrics = a2a_bm.apply_roofline_analysis(
        {"wall_clock_bandwidth_per_chip_gb_s": 75.0}
    )
    self.assertNotIn("peak_ici_bandwidth_per_chip_gb_s", metrics)
    self.assertNotIn("wall_clock_ici_roofline_efficiency_pct", metrics)

  def test_all_to_all_ici_skipped_without_ici_spec(self):
    a2a_bm = self._make_all_to_all_ici_benchmark(
        replica_group=_mock_tpu7x_devices((2, 2, 1)),
        topology_devices=_mock_tpu7x_devices((2, 2, 1)),
        hardware_spec=dataclasses.replace(
            system.TPU7X_HARDWARE_SPEC, ici=None
        ),
    )
    with self.assertLogs(level="WARNING") as logs:
      self.assertIsNone(a2a_bm.get_peak_ici_bandwidth_per_chip())
    self.assertIn("no ICI spec", logs.output[0])

  def test_all_to_all_ici_skipped_for_multi_slice(self):
    slice_0 = _mock_tpu7x_devices((2, 2, 1), slice_index=0)
    slice_1 = _mock_tpu7x_devices((2, 2, 1), slice_index=1)
    a2a_bm = self._make_all_to_all_ici_benchmark(
        replica_group=slice_0 + slice_1,
        topology_devices=slice_0 + slice_1,
    )
    with self.assertLogs(level="WARNING") as logs:
      self.assertIsNone(a2a_bm.get_peak_ici_bandwidth_per_chip())
    self.assertIn("multi-slice", logs.output[0])

  def test_all_to_all_ici_skipped_for_single_chip_group(self):
    topology_devices = _mock_tpu7x_devices((2, 2, 1))
    a2a_bm = self._make_all_to_all_ici_benchmark(
        replica_group=topology_devices[:2],  # Both cores of chip (0, 0, 0).
        topology_devices=topology_devices,
    )
    with self.assertLogs(level="WARNING") as logs:
      self.assertIsNone(a2a_bm.get_peak_ici_bandwidth_per_chip())
    self.assertIn("single chip", logs.output[0])

  def test_all_to_all_ici_skipped_for_scattered_group(self):
    topology_devices = _mock_tpu7x_devices((4, 1, 1), cores=(0,))
    a2a_bm = self._make_all_to_all_ici_benchmark(
        # Chips at x=0 and x=2 only.
        replica_group=[topology_devices[0], topology_devices[2]],
        topology_devices=topology_devices,
    )
    with self.assertLogs(level="WARNING") as logs:
      self.assertIsNone(a2a_bm.get_peak_ici_bandwidth_per_chip())
    self.assertIn("do not form a contiguous 3x1x1 box", logs.output[0])

  def test_replica_groups_ignores_hlo_dump_and_uses_mesh_devices(self):
    """Verify replica groups are derived from mesh.devices instead of HLO dumps."""
    devices = np.array(jax.devices()).reshape((2, 2))

    # Even if an HLO dump file exists with 8-element strided logical indices,
    # _get_first_replica_group_devices derives physical devices
    # directly from self.mesh.devices[indexer].
    dump_dir = self.create_tempdir().full_path
    with open(os.path.join(dump_dir, "after_optimizations.txt"), "w") as f:
      f.write("HloModule ... replica_groups={{0,2,4,6},{1,3,5,7}}")

    # 2x1 sharding on 2x2 mesh ([0, 1], [2, 3]) -> first group is
    # devices[:, 0] = [0, 2] -> parallel.
    config_parallel = collectives.IciCollectivesParams(
        matrix_dim=1024,
        dtype="float32",
        mesh_shape="2x2",
        sharding_strategy="2x1",
    )
    ag_parallel = collectives.AllGatherBenchmark(
        config=config_parallel, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    ag_parallel.setup()
    self.assertEqual(
        [d.id for d in ag_parallel._get_first_replica_group_devices()],  # pylint: disable=protected-access
        [0, 2],
    )
    metrics_p = ag_parallel.calculate_metrics([1.0])
    self.assertEqual(metrics_p["replica_group_type"], "parallel")
    self.assertEqual(metrics_p["replica_group_rank"], 2)

    # Reordered mesh where devices[:, 0] has physical IDs [0, 1] -> non-parallel
    # even when HLO dump file has even logical indices {{0,2},{1,3}}.
    reordered_devices = np.array(
        [devices.flat[0], devices.flat[2], devices.flat[1], devices.flat[3]]
    ).reshape((2, 2))
    reordered_mesh = jax.sharding.Mesh(
        reordered_devices, axis_names=("d_0", "d_1")
    )
    ag_parallel.mesh = reordered_mesh
    self.assertEqual(
        [d.id for d in ag_parallel._get_first_replica_group_devices()],  # pylint: disable=protected-access
        [0, 1],
    )
    metrics_logical = ag_parallel.calculate_metrics([1.0])
    self.assertEqual(metrics_logical["replica_group_type"], "non-parallel")
    self.assertEqual(metrics_logical["replica_group_rank"], 2)

  @parameterized.named_parameters(
      # 2x4x4 misplaced group: ids all even, but half the devices are core 1.
      (
          "even_ids_mixed_cores",
          [0, 2, 4, 6, 8, 10, 12, 14],
          [0, 0, 0, 0, 1, 1, 1, 1],
          "non-parallel",
          6,
      ),
      # Mixed-parity ids that all sit on core 0 -> one core per chip.
      (
          "mixed_ids_same_core",
          [0, 4, 8, 12, 1, 5, 9, 13],
          [0, 0, 0, 0, 0, 0, 0, 0],
          "parallel",
          7,
      ),
  )
  def test_replica_group_type_uses_core_on_chip(
      self, ids, cores, expected_group_type, expected_participating_ranks
  ):
    """Verify parallel detection uses core_on_chip rather than id parity."""
    group = [_MockTpuDevice(id=i, core_on_chip=c) for i, c in zip(ids, cores)]

    config = collectives.IciCollectivesParams(matrix_dim=1024, dtype="float32")
    ag_bm = collectives.AllGatherBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    ag_bm.setup()
    ag_bm._get_first_replica_group_devices = lambda: group  # pylint: disable=protected-access

    metadata = ag_bm.get_workload_metadata()
    self.assertEqual(metadata["replica_group_type"], expected_group_type)
    self.assertLen(ids, metadata["replica_group_rank"])
    tf_multiplier = 2 if expected_group_type == "parallel" else 1
    local_bytes = 1024 * collectives._BASE_N * collectives._BASE_K * 4  # pylint: disable=protected-access
    self.assertEqual(
        metadata["data_transferred_bytes"],
        local_bytes * expected_participating_ranks * tf_multiplier,
    )

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
                "device_placement": collectives.DevicePlacement.ID_SORTED,
            },
            platform_info=test_report_utils.DEFAULT_TEST_PLATFORM_INFO,
            hardware_spec=test_report_utils.DEFAULT_TEST_HARDWARE_SPEC,
        ),
        metrics={
            "shard_size_mib": 32.0,
            "wall_clock_p50_ms": 0.05201,
            "wall_clock_bandwidth_per_chip_gb_s": 350.123,
            "xprof_bandwidth_per_chip_gb_s": 360.000,
            "xprof_p50_ms": 0.04812,
        },
        raw_times_ms=[1.0],
    )
    expected_cols = [
        "dtype",
        "reduce_op",
        "mesh_shape",
        "device_placement",
        "sharding_strategy",
        "matrix_dim",
        "shard_size_mib",
        "wall_clock_p50_ms",
        "wall_clock_bandwidth_per_chip_gb_s",
        "xprof_p50_ms",
        "xprof_bandwidth_per_chip_gb_s",
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
    self.assertIn("id_sorted", table_ar)
    self.assertIn("4096", table_ar)
    self.assertIn("32.00", table_ar)
    self.assertIn("350.12", table_ar)
    self.assertIn("360.00", table_ar)
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
            "shard_size_mib": 64.0,
            "wall_clock_p50_ms": 0.05201,
            "wall_clock_bandwidth_per_chip_gb_s": 350.123,
            "xprof_bandwidth_per_chip_gb_s": 360.000,
            "xprof_p50_ms": 0.04812,
            "peak_ici_bandwidth_per_chip_gb_s": 400.0,
            "wall_clock_ici_roofline_efficiency_pct": 87.53075,
            "xprof_ici_roofline_efficiency_pct": 90.0,
        },
        raw_times_ms=[1.0],
    )
    expected_a2a_cols = [
        "dtype",
        "mesh_shape",
        "device_placement",
        "sharding_strategy",
        "matrix_dim",
        "shard_size_mib",
        "wall_clock_p50_ms",
        "wall_clock_bandwidth_per_chip_gb_s",
        "xprof_p50_ms",
        "xprof_bandwidth_per_chip_gb_s",
        "peak_ici_bandwidth_per_chip_gb_s",
        "wall_clock_ici_roofline_efficiency_pct",
        "xprof_ici_roofline_efficiency_pct",
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
    self.assertIn("360.00", table_a2a)
    self.assertIn("0.0520", table_a2a)
    self.assertIn("0.0481", table_a2a)
    self.assertIn("400.00", table_a2a)
    self.assertIn("87.53", table_a2a)
    self.assertIn("90.00", table_a2a)

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

  def test_dcn_all_reduce_schema_coverage(self):
    """Verify DcnAllReduceBenchmark REPORT_SCHEMA, execution, and output keys."""
    params = {"matrix_dim": 64, "dtype": "float32"}
    config = collectives.DcnAllReduceParams(**params)
    bm = collectives.DcnAllReduceBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    dcn_mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 2)), ("dcn", "ici")
    )
    mock_devices = [
        mock.Mock(
            id=i,
            coords=(i // 2, 0, 0),
            core_on_chip=i % 2,
            slice_index=i // 2,
        )
        for i in range(4)
    ]
    with (
        mock.patch.object(jax, "devices", return_value=mock_devices),
        mock.patch.object(jax.sharding, "Mesh", return_value=dcn_mesh),
    ):
      bm.setup()
    (data,) = bm.generate_inputs()
    self.assertEqual(data.shape, (64, 8, 128))
    out = bm.run_op(data)
    self.assertEqual(out.shape, (64, 8, 128))
    expected = 2.0 * np.array(data)
    for shard in out.addressable_shards:
      np.testing.assert_allclose(np.array(shard.data), expected, rtol=1e-5)
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )
    metrics = bm.derive_chip_metrics(bm.calculate_metrics([1.0]))
    expected_gbps = (
        bm.get_total_bytes() * 8.0 * jax.local_device_count() / (0.001 * 1e9)
    )
    self.assertAlmostEqual(
        metrics["wall_clock_bandwidth_per_host_gbps"], expected_gbps
    )
    self.assertNotIn("wall_clock_bandwidth_per_chip_gb_s", metrics)

  def test_dcn_setup_single_slice_raises_error(self):
    """Verify DcnCollectiveBenchmark.setup rejects fewer than 2 slices."""
    bm = collectives.DcnAllReduceBenchmark(
        config=collectives.DcnAllReduceParams(matrix_dim=64, dtype="float32"),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    # Default CPU devices do not have slice_index -> treated as slice 0 only.
    with self.assertRaisesRegex(
        ValueError, "requires at least 2 slices for DCN collectives"
    ):
      bm.setup()

    # Single-slice TPU devices all having slice_index=0.
    single_slice_devices = [
        mock.Mock(id=i, coords=(i, 0, 0), core_on_chip=0, slice_index=0)
        for i in range(4)
    ]
    with mock.patch.object(jax, "devices", return_value=single_slice_devices):
      with self.assertRaisesRegex(
          ValueError, "requires at least 2 slices for DCN collectives"
      ):
        bm.setup()

  def test_dcn_setup_unequal_slice_device_counts_raises_error(self):
    """Verify DcnCollectiveBenchmark.setup rejects slices with different device counts."""
    bm = collectives.DcnAllReduceBenchmark(
        config=collectives.DcnAllReduceParams(matrix_dim=64, dtype="float32"),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    mismatched_devices = [
        mock.Mock(id=0, coords=(0, 0, 0), core_on_chip=0, slice_index=0),
        mock.Mock(id=1, coords=(0, 0, 0), core_on_chip=1, slice_index=0),
        mock.Mock(id=2, coords=(0, 0, 0), core_on_chip=0, slice_index=1),
    ]
    with mock.patch.object(jax, "devices", return_value=mismatched_devices):
      with self.assertRaisesRegex(
          ValueError, "All slices must have the same number of devices"
      ):
        bm.setup()

  def test_all_gather_schema_coverage(self):
    """Verify AllGatherBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.IciCollectivesParams(**params)
    bm = collectives.AllGatherBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  def test_all_to_all_schema_coverage(self):
    """Verify AllToAllBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.IciCollectivesParams(**params)
    bm = collectives.AllToAllBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  def test_reduce_scatter_schema_coverage(self):
    """Verify ReduceScatterBenchmark REPORT_SCHEMA matches output keys."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.IciCollectivesParams(**params)
    bm = collectives.ReduceScatterBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    test_report_utils.assert_schema_matches_output(
        self, bm, ignored_keys=_COLLECTIVES_IGNORED_KEYS
    )

  @parameterized.named_parameters(
      (
          "collectives",
          collectives.CollectivesParams,
          base.SingleDtypeBenchmarkParams,
          (),
          {"matrix_dim", "seed"},
      ),
      (
          "ici_collectives",
          collectives.IciCollectivesParams,
          collectives.CollectivesParams,
          (),
          {"mesh_shape", "device_placement", "sharding_strategy"},
      ),
      (
          "dcn_collectives",
          collectives.DcnCollectivesParams,
          collectives.CollectivesParams,
          (),
          set(),
      ),
      (
          "all_reduce",
          collectives.AllReduceParams,
          collectives.IciCollectivesParams,
          (collectives.AllReduceParamsMixin,),
          {"reduce_op"},
      ),
      (
          "dcn_all_reduce",
          collectives.DcnAllReduceParams,
          collectives.DcnCollectivesParams,
          (collectives.AllReduceParamsMixin,),
          {"reduce_op"},
      ),
  )
  def test_collectives_params_hierarchy(
      self, param_cls, parent_cls, mixins, expected_new_fields
  ):
    """Verify inheritance and field segregation across collective params."""
    for base_cls in (parent_cls, *mixins):
      self.assertTrue(issubclass(param_cls, base_cls))
    parent_fields = {f.name for f in dataclasses.fields(parent_cls)}
    actual_fields = {f.name for f in dataclasses.fields(param_cls)}
    self.assertEqual(actual_fields, parent_fields | expected_new_fields)
    self.assertFalse(
        issubclass(collectives.AllReduceParamsMixin, base.BaseBenchmarkParams)
    )
    self.assertEqual(
        [f.name for f in dataclasses.fields(collectives.AllReduceParamsMixin)],
        ["reduce_op"],
    )
    self.assertEqual(
        collectives.AllReduceParams.__mro__[1:3],
        (collectives.AllReduceParamsMixin, collectives.IciCollectivesParams),
    )
    self.assertEqual(
        collectives.DcnAllReduceParams.__mro__[1:3],
        (collectives.AllReduceParamsMixin, collectives.DcnCollectivesParams),
    )

  def test_collectives_skips_hbm_roofline_analysis(self):
    """Verify that collectives skip HBM roofline analysis to avoid miscomparing ICI with HBM."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    initial_metrics = {
        "wall_clock_bandwidth_per_chip_gb_s": 95.0,
        "wall_clock_avg_ms": 1.0,
    }
    result_metrics = bm.apply_roofline_analysis(initial_metrics.copy())
    self.assertEqual(result_metrics, initial_metrics)
    self.assertEqual(bm.roofline_mode, constants.RooflineMode.NONE)
    self.assertNotIn("roofline_tflops_limit", result_metrics)
    self.assertNotIn("compute_roofline_efficiency_pct", result_metrics)
    self.assertNotIn("wall_clock_compute_roofline_efficiency_pct", result_metrics)
    self.assertNotIn("peak_hbm_bw_gb_s", result_metrics)
    self.assertNotIn("memory_roofline_efficiency_pct", result_metrics)
    self.assertNotIn("wall_clock_memory_roofline_efficiency_pct", result_metrics)

  def test_zero_latency_and_get_total_bytes(self):
    """Verify zero-latency guard in calculate_throughput_metrics and get_total_bytes."""
    params = {"matrix_dim": 64, "dtype": "bfloat16"}
    config = collectives.AllReduceParams(**params)
    bm = collectives.AllReduceBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    self.assertGreater(bm.get_total_bytes(), 0.0)
    zero_metrics = bm.calculate_throughput_metrics(
        0.0, constants.TimingDomain.WALL_CLOCK
    )
    self.assertEqual(
        zero_metrics["wall_clock_bandwidth_per_chip_gb_s"], float("inf")
    )

  def test_single_rank_replica_group_returns_zero_bandwidth(self):
    """Verify calculate_throughput_metrics returns 0.0 bandwidth when replica_group_rank == 1."""
    config = collectives.AllReduceParams(
        matrix_dim=64,
        dtype="bfloat16",
        mesh_shape="2x2",
        sharding_strategy="1x1",
    )
    bm = collectives.AllReduceBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
    )
    bm.setup()
    metadata = bm.get_workload_metadata()
    self.assertEqual(metadata["replica_group_rank"], 1)
    self.assertEqual(metadata["replica_group_type"], "non-parallel")
    self.assertNotIn("sharding_size", metadata)
    throughput_metrics = bm.calculate_throughput_metrics(
        1.0, constants.TimingDomain.WALL_CLOCK
    )
    self.assertEqual(
        throughput_metrics["wall_clock_bandwidth_per_chip_gb_s"], 0.0
    )
    metrics = bm.calculate_metrics([1.0])
    self.assertEqual(metrics["wall_clock_bandwidth_per_chip_gb_s"], 0.0)
    self.assertNotIn("sharding_size", metrics)

  def test_default_device_placement_uses_mesh_utils(self):
    """Verify default device_placement builds the mesh via mesh_utils."""
    config = collectives.IciCollectivesParams(
        matrix_dim=64,
        dtype="bfloat16",
        mesh_shape="2x2",
        sharding_strategy="2x2",
    )
    self.assertEqual(
        config.device_placement, collectives.DevicePlacement.MESH_UTILS
    )
    bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    with mock.patch.object(
        collectives.mesh_utils,
        "create_device_mesh",
        wraps=collectives.mesh_utils.create_device_mesh,
    ) as mock_create:
      bm.setup()
    mock_create.assert_called_once_with([2, 2], devices=jax.devices())
    self.assertEqual(bm.mesh.devices.shape, (2, 2))

  @parameterized.named_parameters(
      ("full_mesh", "2x2", (256, 8, 128)),
      ("sub_axis", "2x1", (128, 8, 128)),
  )
  def test_id_sorted_device_placement_builds_sorted_mesh(
      self, sharding_strategy, expected_out_shape
  ):
    """Verify id_sorted builds the mesh from devices sorted ascending by id."""
    config = collectives.IciCollectivesParams(
        matrix_dim=64,
        dtype="bfloat16",
        mesh_shape="2x2",
        sharding_strategy=sharding_strategy,
        device_placement="id_sorted",
    )
    bm = collectives.AllGatherBenchmark(
        config=config, hardware_spec=system.TPU7X_HARDWARE_SPEC
    )
    with mock.patch.object(
        collectives.mesh_utils, "create_device_mesh"
    ) as mock_create:
      bm.setup()
    mock_create.assert_not_called()

    self.assertEqual(bm.mesh.devices.shape, (2, 2))
    flat_ids = [d.id for d in bm.mesh.devices.flatten()]
    self.assertEqual(flat_ids, sorted(d.id for d in jax.devices()))

    (data,) = bm.generate_inputs()
    out = bm.run_op(data)
    self.assertEqual(out.shape, expected_out_shape)

  def test_all_reduce_lowering_has_marker(self):
    """Verify the all-reduce op is lowered inside the MARKER named scope."""
    config = collectives.AllReduceParams(matrix_dim=64, dtype="float32")
    bm = collectives.AllReduceBenchmark(
        config=config,
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    bm.setup()
    (x,) = bm.generate_inputs()
    lowered_text = bm._jit_fn.lower(x).as_text(debug_info=True)  # pylint: disable=protected-access
    self.assertIn("MARKER!!!", lowered_text)

  @parameterized.named_parameters(
      (
          "base",
          collectives.BaseCollectiveBenchmark,
          (base.BaseBenchmark,),
          collectives.CollectivesParams,
      ),
      (
          "ici_base",
          collectives.IciCollectiveBenchmark,
          (collectives.BaseCollectiveBenchmark,),
          collectives.IciCollectivesParams,
      ),
      (
          "dcn_base",
          collectives.DcnCollectiveBenchmark,
          (collectives.BaseCollectiveBenchmark,),
          collectives.DcnCollectivesParams,
      ),
      (
          "all_gather",
          collectives.AllGatherBenchmark,
          (collectives.IciCollectiveBenchmark,),
          collectives.IciCollectivesParams,
      ),
      (
          "all_to_all",
          collectives.AllToAllBenchmark,
          (collectives.IciCollectiveBenchmark,),
          collectives.IciCollectivesParams,
      ),
      (
          "reduce_scatter",
          collectives.ReduceScatterBenchmark,
          (collectives.IciCollectiveBenchmark,),
          collectives.ReduceScatterParams,
      ),
      (
          "all_reduce",
          collectives.AllReduceBenchmark,
          (collectives.AllReduceOpMixin, collectives.IciCollectiveBenchmark),
          collectives.AllReduceParams,
      ),
      (
          "dcn_all_reduce",
          collectives.DcnAllReduceBenchmark,
          (collectives.AllReduceOpMixin, collectives.DcnCollectiveBenchmark),
          collectives.DcnAllReduceParams,
      ),
  )
  def test_collective_benchmark_hierarchy(
      self, bm_cls, expected_bases, expected_config
  ):
    """Verify the topology-neutral base, ICI/DCN bases, and the op mixin."""
    self.assertEqual(
        bm_cls.__mro__[1 : 1 + len(expected_bases)], expected_bases
    )
    self.assertIs(bm_cls.Config, expected_config)
    self.assertNotIn(
        "REPORT_SCHEMA", vars(collectives.BaseCollectiveBenchmark)
    )

  @parameterized.named_parameters(
      (
          "all_gather",
          collectives.AllGatherBenchmark,
          [
              "dtype",
              "mesh_shape",
              "device_placement",
              "sharding_strategy",
              "matrix_dim",
              "shard_size_mib",
              "wall_clock_p50_ms",
              "wall_clock_bandwidth_per_chip_gb_s",
              "xprof_p50_ms",
              "xprof_bandwidth_per_chip_gb_s",
          ],
      ),
      (
          "all_to_all",
          collectives.AllToAllBenchmark,
          [
              "dtype",
              "mesh_shape",
              "device_placement",
              "sharding_strategy",
              "matrix_dim",
              "shard_size_mib",
              "wall_clock_p50_ms",
              "wall_clock_bandwidth_per_chip_gb_s",
              "xprof_p50_ms",
              "xprof_bandwidth_per_chip_gb_s",
              "peak_ici_bandwidth_per_chip_gb_s",
              "wall_clock_ici_roofline_efficiency_pct",
              "xprof_ici_roofline_efficiency_pct",
          ],
      ),
      (
          "reduce_scatter",
          collectives.ReduceScatterBenchmark,
          [
              "dtype",
              "mesh_shape",
              "device_placement",
              "sharding_strategy",
              "matrix_dim",
              "shard_size_mib",
              "wall_clock_p50_ms",
              "wall_clock_bandwidth_per_chip_gb_s",
              "xprof_p50_ms",
              "xprof_bandwidth_per_chip_gb_s",
          ],
      ),
      (
          "all_reduce",
          collectives.AllReduceBenchmark,
          [
              "dtype",
              "reduce_op",
              "mesh_shape",
              "device_placement",
              "sharding_strategy",
              "matrix_dim",
              "shard_size_mib",
              "wall_clock_p50_ms",
              "wall_clock_bandwidth_per_chip_gb_s",
              "xprof_p50_ms",
              "xprof_bandwidth_per_chip_gb_s",
          ],
      ),
      (
          "dcn_all_reduce",
          collectives.DcnAllReduceBenchmark,
          [
              "dtype",
              "reduce_op",
              "matrix_dim",
              "shard_size_mib",
              "wall_clock_p50_ms",
              "wall_clock_bandwidth_per_host_gbps",
              "xprof_p50_ms",
              "xprof_bandwidth_per_host_gbps",
          ],
      ),
  )
  def test_report_schema_columns(self, bm_cls, expected_cols):
    """Verify the REPORT_SCHEMA columns of every registered collective."""
    self.assertEqual([col for col, _ in bm_cls.REPORT_SCHEMA], expected_cols)

  @parameterized.named_parameters(
      ("all_gather", collectives.AllGatherBenchmark),
      ("all_to_all", collectives.AllToAllBenchmark),
      ("reduce_scatter", collectives.ReduceScatterBenchmark),
  )
  def test_get_run_identifier(self, bm_cls):
    """Verify non-all-reduce collectives use the dim_<matrix_dim> identifier."""
    bm = bm_cls(
        config=collectives.IciCollectivesParams(matrix_dim=64),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    self.assertEqual(bm.get_run_identifier(), "dim_64")

  def test_all_reduce_op_mixin_requires_reduce_op_config(self):
    """Verify AllReduceOpMixin rejects a config without AllReduceParamsMixin."""

    class _Bad(
        collectives.AllReduceOpMixin[collectives.IciCollectivesParams],
        collectives.IciCollectiveBenchmark[collectives.IciCollectivesParams],
    ):
      pass

    bm = _Bad(
        config=collectives.IciCollectivesParams(matrix_dim=64),
        hardware_spec=system.TPU7X_HARDWARE_SPEC,
        mesh=self.mock_mesh,
    )
    with self.assertRaises(TypeError):
      bm._get_reduce_op()  # pylint: disable=protected-access


class CollectivesParamsValidationTest(parameterized.TestCase):
  """Verifies the bounds declared on CollectivesParams fields."""

  @parameterized.parameters(
      ("matrix_dim", 0, 1),
      ("matrix_dim", -1, 1),
      ("seed", -1, 0),
  )
  def test_out_of_range_values_raise_error(self, field, value, bound):
    with self.assertRaisesRegex(ValueError, f"{field} must be >= {bound}"):
      collectives.CollectivesParams(**{field: value})

  def test_reduce_op_coerces_to_enum(self):
    params = collectives.AllReduceParams(reduce_op="mean")
    self.assertEqual(params.reduce_op, collectives.ReduceOp.MEAN)
    self.assertIsInstance(params.reduce_op, collectives.ReduceOp)

  def test_invalid_reduce_op_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Invalid reduce_op 'unsupported'"):
      collectives.AllReduceParams(reduce_op="unsupported")

  @parameterized.named_parameters(
      ("mesh_utils", "mesh_utils", collectives.DevicePlacement.MESH_UTILS),
      ("id_sorted", "id_sorted", collectives.DevicePlacement.ID_SORTED),
  )
  def test_device_placement_coerces_to_enum(self, raw, expected):
    """Verify YAML string values are coerced into DevicePlacement members."""
    params = collectives.IciCollectivesParams(device_placement=raw)
    self.assertEqual(params.device_placement, expected)
    self.assertIsInstance(params.device_placement, collectives.DevicePlacement)
    self.assertEqual(dataclasses.asdict(params)["device_placement"], raw)

  def test_invalid_device_placement_raises_error(self):
    with self.assertRaisesRegex(ValueError, "Invalid device_placement 'bogus'"):
      collectives.IciCollectivesParams(device_placement="bogus")

  @parameterized.parameters(
      "mesh_shape", "device_placement", "sharding_strategy"
  )
  def test_api_change_collectives_params_rejects_mesh_fields(self, field):
    """Verify ICI mesh fields live on IciCollectivesParams only."""
    with self.assertRaises(TypeError):
      collectives.CollectivesParams(**{field: "2x2"})

  @parameterized.named_parameters(
      ("all_reduce", collectives.AllReduceParams, frozenset()),
      ("ici", collectives.IciCollectivesParams, frozenset({"reduce_op"})),
      (
          "dcn_all_reduce",
          collectives.DcnAllReduceParams,
          frozenset({"mesh_shape", "device_placement", "sharding_strategy"}),
      ),
      (
          "dcn",
          collectives.DcnCollectivesParams,
          frozenset({
              "mesh_shape",
              "device_placement",
              "sharding_strategy",
              "reduce_op",
          }),
      ),
  )
  def test_field_defaults_and_help_unchanged(self, params_cls, excluded):
    """Verify every collective field keeps its default and metadata, by name."""
    fields = {f.name: f for f in dataclasses.fields(params_cls)}
    self.assertEqual(set(fields), set(_HEAD_FIELD_SPEC) - excluded)
    for name, field in fields.items():
      default, metadata = _HEAD_FIELD_SPEC[name]
      self.assertEqual(field.default, default, msg=name)
      self.assertEqual(dict(field.metadata), metadata, msg=name)

  @parameterized.named_parameters(
      (op.name.lower(), op.value) for op in collectives.ReduceOp
  )
  def test_all_reduce_reduce_op_help_message_and_default_value(
      self, choice: str
  ):
    """Verifies help message includes choice and matching default."""
    parser = argparse.ArgumentParser()
    cli.add_dataclass_arguments(parser, collectives.AllReduceParams)
    s = io.StringIO()
    parser.print_help(file=s)
    help_msg = s.getvalue()

    normalized_help = " ".join(help_msg.split())
    self.assertIn(
        choice,
        help_msg,
        f"Enum value '{choice}' missing from help message: {help_msg}",
    )

    default_op = collectives.AllReduceParams().reduce_op.value
    self.assertIn(
        f"(default: {default_op})",
        normalized_help,
        f"Displayed default '{default_op}' missing from help: {help_msg}",
    )


class AllToAllPeakBandwidthTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # Replica group spans the full ICI topology.
      # M = 2 (wrap) * 4*4 = 32: 4*100*32*63/64^2.
      ("tpu7x_4x4x4_parallel", 64, 1, (4, 4, 4), (4, 4, 4), _ALL_DIMS_GE_4,
       196.875),
      ("tpu7x_4x4x4_non_parallel", 128, 2, (4, 4, 4), (4, 4, 4),
       _ALL_DIMS_GE_4, 196.875),
      # M = 2 * 8*8 = 128.
      ("tpu7x_8x8x8_parallel", 512, 1, (8, 8, 8), (8, 8, 8), _ALL_DIMS_GE_4,
       99.8046875),
      # No wrap (2 < 4); cut across a 4-axis: M = 2*4 = 8 (two smaller dims).
      ("tpu7x_2x4x4_parallel", 32, 1, (2, 4, 4), (2, 4, 4), _ALL_DIMS_GE_4,
       96.875),
      # M = 2.
      ("tpu7x_2x2x1_parallel", 4, 1, (2, 2, 1), (2, 2, 1), _ALL_DIMS_GE_4,
       150.0),
      ("tpu7x_2x2x1_non_parallel", 8, 2, (2, 2, 1), (2, 2, 1),
       _ALL_DIMS_GE_4, 150.0),
      # v6e: no wrap below 16; M = 4.
      ("v6e_4x4", 16, 1, (4, 4, 1), (4, 4, 1), _PER_DIM_GE_16, 93.75),
      # Both axes wrap: M = 2 * 16 = 32.
      ("v6e_16x16", 256, 1, (16, 16, 1), (16, 16, 1), _PER_DIM_GE_16,
       49.8046875),
      # Only the 16-axis wraps: cut across it gives M = 2 * 8 = 16, and the
      # cut across the 8-axis gives 1 * 16 = 16.
      ("v6e_16x8", 128, 1, (16, 8, 1), (16, 8, 1), _PER_DIM_GE_16,
       49.609375),
      # No wrap-around rule: M = 16.
      ("no_wrap_4x4x4", 64, 1, (4, 4, 4), (4, 4, 4), _NO_WRAP, 98.4375),
      # Replica group spans a subset of the ICI topology.
      # 4x4 plane of a 4x4x4 cube: spans x and y fully, M = 2 * 4 = 8.
      ("tpu7x_4x4_in_4x4x4", 16, 1, (4, 4, 1), (4, 4, 4), _ALL_DIMS_GE_4,
       187.5),
      # 4-chip ring along x of a 4x4x4 cube: M = 2 * 1 = 2.
      ("tpu7x_4x1x1_in_4x4x4", 4, 1, (4, 1, 1), (4, 4, 4), _ALL_DIMS_GE_4,
       150.0),
      # x is not fully spanned (no wrap): x cut 1*16 = 16; y/z cuts 2*(2*4)
      # = 16; M = 16.
      ("tpu7x_2x4x4_in_4x4x4", 32, 1, (2, 4, 4), (4, 4, 4), _ALL_DIMS_GE_4,
       193.75),
      # Topology without wrap-around: M = 4.
      ("tpu7x_4x4_in_4x4x1", 16, 1, (4, 4, 1), (4, 4, 1), _ALL_DIMS_GE_4,
       93.75),
  )
  def test_peak_bandwidth(
      self, rank, k, group_dims, topology_dims, rule, expected
  ):
    self.assertAlmostEqual(
        _peak(rank, k, group_dims, topology_dims, rule), expected
    )

  @parameterized.parameters(
      ((2, 2, 1),), ((2, 4, 4),), ((4, 4, 4),), ((8, 8, 8),)
  )
  def test_parallel_and_non_parallel_peaks_match(self, dims):
    """The same chips move the same bytes over the same links."""
    chips = math.prod(dims)
    parallel = _peak(chips, 1, dims, dims, _ALL_DIMS_GE_4)
    non_parallel = _peak(2 * chips, 2, dims, dims, _ALL_DIMS_GE_4)
    self.assertAlmostEqual(parallel, non_parallel)

  def test_duplicate_chip_coords_are_deduplicated(self):
    """Both cores of a chip share coordinates and count as one chip."""
    coords = _box_coords((2, 2, 1))
    peak = collectives.all_to_all_peak_bandwidth_per_chip_gb_s(
        replica_group_rank=8,
        devices_per_chip_in_replica_group=2,
        replica_group_chip_coords=coords + coords,
        topology_dims=(2, 2, 1),
        link_bw_per_direction_gb_s=_LINK_BW,
        wraparound_rule=_ALL_DIMS_GE_4,
    )
    self.assertAlmostEqual(peak, 150.0)

  def test_single_chip_raises(self):
    with self.assertRaisesRegex(ValueError, "single chip"):
      collectives.all_to_all_peak_bandwidth_per_chip_gb_s(
          replica_group_rank=2,
          devices_per_chip_in_replica_group=2,
          replica_group_chip_coords=[(0, 0, 0), (0, 0, 0)],
          topology_dims=(2, 2, 1),
          link_bw_per_direction_gb_s=_LINK_BW,
          wraparound_rule=_ALL_DIMS_GE_4,
      )

  def test_scattered_replica_group_raises(self):
    with self.assertRaisesRegex(
        ValueError, "2 chips do not form a contiguous 3x1x1 box"
    ):
      collectives.all_to_all_peak_bandwidth_per_chip_gb_s(
          replica_group_rank=2,
          devices_per_chip_in_replica_group=1,
          replica_group_chip_coords=[(0, 0, 0), (2, 0, 0)],
          topology_dims=(4, 2, 1),
          link_bw_per_direction_gb_s=_LINK_BW,
          wraparound_rule=_ALL_DIMS_GE_4,
      )

  def test_empty_replica_group_raises(self):
    with self.assertRaisesRegex(ValueError, "no chip coordinates"):
      collectives.all_to_all_peak_bandwidth_per_chip_gb_s(
          replica_group_rank=0,
          devices_per_chip_in_replica_group=1,
          replica_group_chip_coords=[],
          topology_dims=(2, 2, 1),
          link_bw_per_direction_gb_s=_LINK_BW,
          wraparound_rule=_ALL_DIMS_GE_4,
      )

  def test_asymmetric_replica_group_raises(self):
    """Both cores of one chip plus one core of the other 3 chips (R=5)."""
    with self.assertRaisesRegex(
        ValueError, "5 devices, not 4 chips x 2 devices per chip"
    ):
      collectives.all_to_all_peak_bandwidth_per_chip_gb_s(
          replica_group_rank=5,
          devices_per_chip_in_replica_group=2,
          replica_group_chip_coords=[(0, 0, 0)] + _box_coords((2, 2, 1)),
          topology_dims=(2, 2, 1),
          link_bw_per_direction_gb_s=_LINK_BW,
          wraparound_rule=_ALL_DIMS_GE_4,
      )


if __name__ == "__main__":
  absltest.main()
