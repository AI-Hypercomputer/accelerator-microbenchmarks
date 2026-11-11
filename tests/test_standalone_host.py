"""Unit tests for standalone_host.py."""

import io
import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import standalone_host


def _flags(args: str) -> dict[str, str | None]:
  """Parses a LIBTPU_INIT_ARGS string into a name -> value mapping."""
  parsed = {}
  for token in args.split():
    name, sep, value = token.partition("=")
    parsed[name] = value if sep else None
  return parsed


class StandaloneHostTest(parameterized.TestCase):
  """Unit tests for configure_standalone_host()."""

  _EXPECTED_FLAGS = {
      "--deepsea_hal_test_skip_slicebuilder": "true",
      "--deepsea_hal_test_allow_multichip_skip_slicebuilder": "true",
      "--xla_tpu_use_enhanced_launch_barrier": "false",
      "--xla_tpu_enable_offloading_copy_to_sparsecore": "false",
  }

  def _configure(self, env: dict[str, str]) -> tuple[dict[str, str], str]:
    """Runs configure_standalone_host() in an isolated env; returns env, log."""
    with mock.patch.dict(os.environ, env, clear=True):
      with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
        standalone_host.configure_standalone_host()
      return dict(os.environ), fake_out.getvalue()

  def test_sets_standalone_host_topology_env(self):
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

  def test_removes_tpu_process_addresses_env(self):
    """GKE multi-host process addresses are removed so JAX cannot pick them up.

    `GkeTpuCluster` checks `TPU_PROCESS_ADDRESSES(_PATH)` before
    `TPU_WORKER_HOSTNAMES`, so overriding the latter alone is not enough.
    """
    env, log = self._configure({
        "TPU_PROCESS_ADDRESSES": "10.0.0.1:8476,10.0.0.2:8476",
        "TPU_PROCESS_ADDRESSES_PATH": "/tmp/tpu_process_addresses",
    })
    self.assertNotIn("TPU_PROCESS_ADDRESSES", env)
    self.assertNotIn("TPU_PROCESS_ADDRESSES_PATH", env)
    self.assertIn(
        "unset TPU_PROCESS_ADDRESSES (was '10.0.0.1:8476,10.0.0.2:8476')", log
    )
    self.assertIn(
        "unset TPU_PROCESS_ADDRESSES_PATH (was '/tmp/tpu_process_addresses')",
        log,
    )

  def test_absent_tpu_process_addresses_env_is_not_logged(self):
    """Nothing is logged for env vars that were not set to begin with."""
    _, log = self._configure({})
    self.assertNotIn("unset", log)

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
        os.environ,
        {
            "LIBTPU_INIT_ARGS": "--xla_tpu_foo=1",
            "TPU_PROCESS_ADDRESSES": "10.0.0.1:8476",
        },
        clear=True,
    ):
      with mock.patch.object(sys, "stdout", new=io.StringIO()):
        standalone_host.configure_standalone_host()
        first = dict(os.environ)
      with mock.patch.object(sys, "stdout", new=io.StringIO()) as fake_out:
        standalone_host.configure_standalone_host()
      self.assertEqual(dict(os.environ), first)
      self.assertEqual(fake_out.getvalue(), "")


if __name__ == "__main__":
  absltest.main()
