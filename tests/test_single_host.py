"""Unit tests for single_host.py."""

import io
import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import single_host


def _flags(args: str) -> dict[str, str | None]:
  """Parses a LIBTPU_INIT_ARGS string into a name -> value mapping."""
  parsed = {}
  for token in args.split():
    name, sep, value = token.partition("=")
    parsed[name] = value if sep else None
  return parsed


class SingleHostTest(parameterized.TestCase):
  """Unit tests for configure_single_host_tpu()."""

  _EXPECTED_FLAGS = {
      "--deepsea_hal_test_skip_slicebuilder": "true",
      "--deepsea_hal_test_allow_multichip_skip_slicebuilder": "true",
      "--xla_tpu_use_enhanced_launch_barrier": "false",
  }

  def _configure(self, env: dict[str, str]) -> tuple[dict[str, str], str]:
    """Runs configure_single_host_tpu() in an isolated env; returns env, log."""
    with mock.patch.dict(os.environ, env, clear=True):
      with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
        single_host.configure_single_host_tpu()
      return dict(os.environ), fake_out.getvalue()

  def test_sets_single_host_topology_env(self):
    """Forces a 1-host topology and skips the metadata server query."""
    env, _ = self._configure({})
    self.assertEqual(env["TPU_HOST_BOUNDS"], "1,1,1")
    self.assertEqual(env["TPU_WORKER_ID"], "0")
    self.assertEqual(env["TPU_WORKER_HOSTNAMES"], "localhost")
    self.assertEqual(env["TPU_SKIP_MDS_QUERY"], "true")
    self.assertEqual(_flags(env["LIBTPU_INIT_ARGS"]), self._EXPECTED_FLAGS)

  def test_overrides_injected_multi_host_env_and_logs_previous_value(self):
    """GKE-injected multi-host values are replaced and the change is logged."""
    env, log = self._configure({
        "TPU_WORKER_ID": "2",
        "TPU_WORKER_HOSTNAMES": "host-0,host-1,host-2,host-3",
        "TPU_HOST_BOUNDS": "2,2,1",
    })
    self.assertEqual(env["TPU_WORKER_ID"], "0")
    self.assertEqual(env["TPU_WORKER_HOSTNAMES"], "localhost")
    self.assertEqual(env["TPU_HOST_BOUNDS"], "1,1,1")
    self.assertIn("override TPU_WORKER_ID=0 (was '2')", log)
    self.assertIn("override TPU_HOST_BOUNDS=1,1,1 (was '2,2,1')", log)

  def test_preserves_existing_libtpu_flags(self):
    """Flags from op_flags.yaml are kept, in order, ahead of the additions."""
    env, _ = self._configure(
        {"LIBTPU_INIT_ARGS": "--xla_tpu_foo=1 --xla_tpu_bar --xla_tpu_baz=x"}
    )
    args = env["LIBTPU_INIT_ARGS"]
    self.assertTrue(
        args.startswith("--xla_tpu_foo=1 --xla_tpu_bar --xla_tpu_baz=x ")
    )
    self.assertEqual(
        _flags(args),
        {
            "--xla_tpu_foo": "1",
            "--xla_tpu_bar": None,
            "--xla_tpu_baz": "x",
            **self._EXPECTED_FLAGS,
        },
    )

  @parameterized.named_parameters(
      ("lowercase_true", "--xla_tpu_use_enhanced_launch_barrier=true"),
      ("capitalised_true", "--xla_tpu_use_enhanced_launch_barrier=True"),
      ("numeric_true", "--xla_tpu_use_enhanced_launch_barrier=1"),
      ("bare_flag", "--xla_tpu_use_enhanced_launch_barrier"),
  )
  def test_overrides_launch_barrier_flag_regardless_of_spelling(self, flag):
    """Any pre-set launch-barrier value is overridden in place, not appended."""
    env, log = self._configure(
        {"LIBTPU_INIT_ARGS": f"--xla_tpu_foo=1 {flag} --xla_tpu_bar=2"}
    )
    args = env["LIBTPU_INIT_ARGS"]
    self.assertEqual(
        args.split()[:3],
        [
            "--xla_tpu_foo=1",
            "--xla_tpu_use_enhanced_launch_barrier=false",
            "--xla_tpu_bar=2",
        ],
    )
    self.assertEqual(args.count("--xla_tpu_use_enhanced_launch_barrier"), 1)
    self.assertIn("override LIBTPU_INIT_ARGS=", log)

  def test_idempotent(self):
    """A second call changes nothing and does not duplicate flags."""
    with mock.patch.dict(
        os.environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_foo=1"}, clear=True
    ):
      with mock.patch.object(sys, "stdout", new=io.StringIO()):
        single_host.configure_single_host_tpu()
        first = dict(os.environ)
      with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
        single_host.configure_single_host_tpu()
      self.assertEqual(dict(os.environ), first)
      self.assertEqual(fake_out.getvalue(), "")

  def test_supported_benchmarks_are_registered_names(self):
    """The allow-list only names benchmarks (or aliases) that exist."""
    benchmark_loader.load_all_benchmarks()
    registered = set(
        registry.benchmark_registry.list_benchmark_names(
            include_experimental=True, include_aliases=True
        )
    )
    self.assertContainsSubset(single_host.SUPPORTED_BENCHMARKS, registered)


if __name__ == "__main__":
  absltest.main()
