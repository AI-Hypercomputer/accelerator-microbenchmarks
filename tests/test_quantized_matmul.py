"""Unit tests for kernels/quantized_matmul.py (Pallas interpret mode on CPU)."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from accelerator_microbenchmarks.core import dtypes
from accelerator_microbenchmarks.kernels import quantized_matmul
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np


def setUpModule():  # pylint: disable=invalid-name
  jax.config.update("jax_platform_name", "cpu")


def _reference_subchannel_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    sc_size: int,
    quant_dtype: jax.typing.DTypeLike,
    rhs_quant_dtype: jax.typing.DTypeLike | None = None,
    out_dtype: jax.typing.DTypeLike | None = None,
    dynamic: bool = True,
    dequantize: bool = True,
) -> jax.Array:
  """Pure-JAX sub-channel quantized GEMM reference for numerical parity."""
  m, k = lhs.shape
  _, n = rhs.shape
  num_sc = k // sc_size
  lhs_q_dt = jnp.dtype(quant_dtype)
  rhs_q_dt = lhs_q_dt if rhs_quant_dtype is None else jnp.dtype(rhs_quant_dtype)
  out_dt = lhs.dtype if out_dtype is None else jnp.dtype(out_dtype)

  def _quant_blocks(x_3d, q_dt, reduce_axis):
    x_f32 = x_3d.astype(jnp.float32)
    q_max = dtypes.get_dtype_max(q_dt)
    if dynamic:
      amax = jnp.max(jnp.abs(x_f32), axis=reduce_axis, keepdims=True)
      clamped_amax = jnp.maximum(amax, 1e-7)
      scale = clamped_amax / q_max
      inv_scale = q_max / clamped_amax
    else:
      scale_shape = list(x_3d.shape)
      scale_shape[reduce_axis] = 1
      scale = jnp.full(scale_shape, 0.5, dtype=jnp.float32)
      inv_scale = 2.0
    q = x_f32 * inv_scale
    if jnp.issubdtype(q_dt, jnp.integer):
      q = jnp.clip(jnp.round(q), -q_max, q_max)
    else:
      q = jnp.clip(q, -q_max, q_max)
    return q.astype(q_dt), scale

  qa, sf_a = _quant_blocks(
      lhs.reshape(m, num_sc, sc_size), lhs_q_dt, reduce_axis=2
  )
  qb, sf_b = _quant_blocks(
      rhs.reshape(num_sc, sc_size, n), rhs_q_dt, reduce_axis=1
  )
  both_int = jnp.issubdtype(lhs_q_dt, jnp.integer) and jnp.issubdtype(
      rhs_q_dt, jnp.integer
  )
  dot_acc_dt = jnp.int32 if both_int else jnp.float32

  acc = jnp.zeros((m, n), dtype=jnp.float32)
  for g in range(num_sc):
    dot = jax.lax.dot_general(
        qa[:, g, :],
        qb[g, :, :],
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=dot_acc_dt,
    ).astype(jnp.float32)
    if dequantize:
      dot = dot * (sf_a[:, g, :] * sf_b[g, :, :])
    acc = acc + dot
  return acc.astype(out_dt)


class QuantizedMatmulBlocksTest(parameterized.TestCase):
  """Tests hardware-spec VMEM tile selection and explicit block validation."""

  def setUp(self):
    super().setUp()
    self._mock_get_tpu_info = self.enter_context(
        mock.patch.object(
            pltpu,
            "get_tpu_info",
            autospec=True,
            return_value=pltpu.get_tpu_info_for_chip(
                pltpu.ChipVersion.TPU_7X, 1
            ),
        )
    )

  def test_default_vmem_budget_and_mxu_width_read_tpu_info(self):
    """Verifies hardware VMEM capacity and MXU width are queried on TPU."""
    self.assertEqual(quantized_matmul.default_vmem_budget_bytes(), 64 << 20)
    self._mock_get_tpu_info.return_value = pltpu.get_tpu_info_for_chip(
        pltpu.ChipVersion.TPU_V5E, 1
    )
    self.assertEqual(quantized_matmul.default_vmem_budget_bytes(), 128 << 20)
    tile_128_mxu = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        1024,
        16384,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
    )
    self.assertEqual(tile_128_mxu[0], 1024)

  def test_default_blocks_scale_up_across_tpu_vmem_generations(self):
    """Verifies larger hardware VMEM selects larger reuse tiles."""
    tile_64m = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        16384,
        16384,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=64 << 20,
    )
    tile_128m = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        16384,
        16384,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=128 << 20,
    )
    tile_256m = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        16384,
        16384,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=256 << 20,
    )
    area_64m = tile_64m[0] * tile_64m[2]
    area_128m = tile_128m[0] * tile_128m[2]
    area_256m = tile_256m[0] * tile_256m[2]
    self.assertLess(area_64m, area_128m)
    self.assertLess(area_128m, area_256m)
    self.assertEqual(tile_64m, (1024, 1024, 2048))

  def test_default_blocks_bounds_m_tile_for_vmem_headroom(self):
    """Verifies `block_m` stays within `8 * mxu_col` including small IO."""
    tile = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        1024,
        16384,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=256 << 20,
    )
    self.assertEqual(tile, (2048, 1024, 4096))
    tall_skinny_tile = quantized_matmul.default_quantized_matmul_blocks(
        8192,
        256,
        128,
        sc_size=128,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=256 << 20,
    )
    self.assertLessEqual(tall_skinny_tile[0], 2048)

  def test_default_blocks_non_power_of_two_sc_size(self):
    """Verifies selector splits K on non-power-of-2 sc_size multiples of 128."""
    bm, bk, bn = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        12288,
        12288,
        sc_size=384,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=64 << 20,
    )
    self.assertLess(bk, 12288)
    self.assertEqual(bk % 384, 0)
    self.assertGreaterEqual(bn, 1024)
    quantized_matmul.validate_quantized_matmul_blocks(
        4096,
        12288,
        12288,
        sc_size=384,
        block_m=bm,
        block_k=bk,
        block_n=bn,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=64 << 20,
    )

  def test_default_blocks_non_power_of_two_llm_n_preserves_mxu_col_width(self):
    """Verifies non-power-of-2 N keeps mxu_col-aligned block_n and strips."""
    bm, bk, bn = quantized_matmul.default_quantized_matmul_blocks(
        4096,
        4096,
        11008,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
    )
    self.assertEqual(11008 % bn, 0)
    self.assertEqual(bn % 256, 0)
    compute_tile_n = quantized_matmul._select_compute_tile_n(  # pylint: disable=protected-access
        bm, bn
    )
    self.assertEqual(bn % compute_tile_n, 0)
    self.assertEqual(compute_tile_n % 256, 0)
    quantized_matmul.validate_quantized_matmul_blocks(
        4096,
        4096,
        11008,
        sc_size=512,
        block_m=bm,
        block_k=bk,
        block_n=bn,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
    )

    bm2, bk2, bn2 = quantized_matmul.default_quantized_matmul_blocks(
        8192,
        7168,
        18432,
        sc_size=512,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=64 << 20,
    )
    self.assertEqual(bn2, 1536)
    self.assertEqual(bn2 % 256, 0)
    self.assertEqual(
        quantized_matmul._select_compute_tile_n(bm2, bn2),  # pylint: disable=protected-access
        512,
    )
    quantized_matmul.validate_quantized_matmul_blocks(
        8192,
        7168,
        18432,
        sc_size=512,
        block_m=bm2,
        block_k=bk2,
        block_n=bn2,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=64 << 20,
    )

  @parameterized.named_parameters(
      (
          "non_positive_block",
          dict(block_m=0, block_k=128, block_n=128),
          "must be positive",
      ),
      (
          "non_dividing_block",
          dict(block_m=128, block_k=384, block_n=128),
          "must divide matrix dimensions",
      ),
      (
          "block_k_not_multiple_of_sc_size",
          dict(block_m=128, block_k=128, block_n=128, sc_size=256),
          "divisible by sc_size",
      ),
      (
          "sc_size_not_lane_aligned",
          dict(sc_size=64, block_m=128, block_k=256, block_n=128),
          "sc_size=64 must be a multiple of lanes=128",
      ),
      (
          "unaligned_block_n",
          dict(block_m=128, block_k=128, block_n=64),
          "multiple of lanes",
      ),
      (
          "vmem_overflow",
          dict(
              m=4096,
              k=4096,
              n=4096,
              block_m=4096,
              block_k=4096,
              block_n=4096,
              vmem_budget_bytes=64 << 20,
          ),
          "exceeding usable budget",
      ),
      (
          "sc_size_ge_k",
          dict(sc_size=512, block_m=128, block_k=512, block_n=128),
          "0 < sc_size < k",
      ),
  )
  def test_validate_blocks_rejects_unusable_inputs(self, overrides, err_regex):
    """Verifies explicit unusable block sizes raise descriptive ValueError."""
    kwargs = dict(
        m=256,
        k=512,
        n=256,
        sc_size=128,
        block_m=128,
        block_k=256,
        block_n=128,
        in_dtype=jnp.bfloat16,
        out_dtype=jnp.bfloat16,
        vmem_budget_bytes=64 << 20,
    )
    kwargs.update(overrides)
    with self.assertRaisesRegex(ValueError, err_regex):
      quantized_matmul.validate_quantized_matmul_blocks(**kwargs)


class QuantizedMatmulKernelTest(parameterized.TestCase):
  """Tests Pallas `quantized_matmul` execution in CPU interpret mode."""

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.object(
            pltpu,
            "get_tpu_info",
            autospec=True,
            return_value=pltpu.get_tpu_info_for_chip(
                pltpu.ChipVersion.TPU_7X, 1
            ),
        )
    )

  def _assert_matches_reference(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      *,
      sc_size: int = 128,
      block_m: int = 64,
      block_k: int = 256,
      block_n: int = 128,
      quant_dtype: jax.typing.DTypeLike = "float8_e4m3fn",
      rhs_quant_dtype: jax.typing.DTypeLike | None = None,
      out_dtype: jax.typing.DTypeLike = jnp.float32,
      dynamic: bool = True,
      dequantize: bool = True,
      exact: bool = False,
  ) -> None:
    """Asserts Pallas interpret output matches `_reference_subchannel_matmul`."""
    actual = quantized_matmul.quantized_matmul(
        lhs,
        rhs,
        sc_size=sc_size,
        block_m=block_m,
        block_k=block_k,
        block_n=block_n,
        quant_dtype=quant_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=out_dtype,
        dynamic=dynamic,
        dequantize=dequantize,
        interpret=True,
    )
    expected = _reference_subchannel_matmul(
        lhs,
        rhs,
        sc_size=sc_size,
        quant_dtype=quant_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=out_dtype,
        dynamic=dynamic,
        dequantize=dequantize,
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(actual))))
    if exact:
      np.testing.assert_array_equal(
          np.asarray(actual, dtype=np.float32),
          np.asarray(expected, dtype=np.float32),
      )
      return
    tol = 1e-2 if out_dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=tol,
        atol=tol,
    )

  @parameterized.named_parameters(
      ("fp8_dynamic_dequant", "float8_e4m3fn", None, jnp.float32, True, True),
      (
          "fp8_dynamic_dequant_bf16_out",
          "float8_e4m3fn",
          None,
          jnp.bfloat16,
          True,
          True,
      ),
      ("fp8_static_dequant", "float8_e4m3fn", None, jnp.float32, False, True),
      (
          "fp8_dynamic_no_dequant",
          "float8_e4m3fn",
          None,
          jnp.float32,
          True,
          False,
      ),
      (
          "e5m2_int4_dynamic_dequant",
          "float8_e5m2",
          "int4",
          jnp.float32,
          True,
          True,
      ),
      (
          "fp8_int4_static_dequant",
          "float8_e4m3fn",
          "int4",
          jnp.float32,
          False,
          True,
      ),
      ("int8_dynamic_dequant", "int8", None, jnp.float32, True, True),
  )
  def test_subchannel_kernel_matches_reference(
      self, quant_dtype, rhs_quant_dtype, out_dtype, dynamic, dequantize
  ):
    """Verifies multi-tile sub-channel Pallas kernel matches pure-JAX ref."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(42))
    lhs = jax.random.normal(key_a, (128, 512), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key_b, (512, 256), dtype=jnp.bfloat16)
    self._assert_matches_reference(
        lhs,
        rhs,
        quant_dtype=quant_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
        out_dtype=out_dtype,
        dynamic=dynamic,
        dequantize=dequantize,
    )

  def test_subchannel_kernel_multi_n_strip_matches_reference(self):
    """Verifies steps_n == 2 multi-strip indexing matches pure-JAX reference."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(7))
    lhs = jax.random.normal(key_a, (128, 256), dtype=jnp.bfloat16)
    rhs = jax.random.normal(key_b, (256, 256), dtype=jnp.bfloat16)
    with mock.patch.object(
        quantized_matmul,
        "_select_compute_tile_n",
        autospec=True,
        return_value=128,
    ):
      self._assert_matches_reference(lhs, rhs, block_n=256)

  def test_subchannel_kernel_float32_inputs_match_reference(self):
    """Verifies float32 inputs match the pure-JAX sub-channel reference."""
    key_a, key_b = jax.random.split(jax.random.PRNGKey(13))
    lhs = jax.random.normal(key_a, (128, 512), dtype=jnp.float32)
    rhs = jax.random.normal(key_b, (512, 256), dtype=jnp.float32)
    self._assert_matches_reference(lhs, rhs)

  def test_subchannel_kernel_all_zeros_clamp_matches_reference(self):
    """Verifies all-zero inputs exercise the 1e-7 amax clamp without NaNs."""
    lhs = jnp.zeros((64, 256), dtype=jnp.bfloat16)
    rhs = jnp.zeros((256, 128), dtype=jnp.bfloat16)
    self._assert_matches_reference(lhs, rhs, exact=True)

  def test_subchannel_kernel_static_outlier_clipping_matches_reference(self):
    """Verifies static scaling clips outliers exceeding 0.5 * q_max."""
    # Values exceed 0.5 * q_max (224.0 for float8_e4m3fn, 3.5 for int4) and
    # must clip to [-q_max, q_max] without producing NaNs or wraparound.
    lhs_pattern = jnp.array([[500.0, -500.0, 112.0, 0.0]], dtype=jnp.bfloat16)
    rhs_pattern = jnp.array([[10.0, -10.0, 1.0, 0.0]], dtype=jnp.bfloat16)
    lhs = jnp.tile(lhs_pattern, (64, 64))
    rhs = jnp.tile(rhs_pattern, (256, 32))
    self._assert_matches_reference(
        lhs, rhs, rhs_quant_dtype="int4", dynamic=False
    )

  @parameterized.named_parameters(
      ("fp8_e4m3_same", jnp.float8_e4m3fn, jnp.float8_e4m3fn),
      ("fp8_e5m2_same", jnp.float8_e5m2, jnp.float8_e5m2),
      ("int8_same", jnp.int8, jnp.int8),
      ("fp8_e4m3_int4", jnp.float8_e4m3fn, jnp.int4),
      ("fp8_e5m2_int4", jnp.float8_e5m2, jnp.int4),
  )
  def test_validate_quant_dtype_pair_valid(self, quant_dt, rhs_quant_dt):
    """Verifies valid (quant_dtype, rhs_quant_dtype) pairs are accepted."""
    quantized_matmul.validate_quant_dtype_pair(quant_dt, rhs_quant_dt)

  @parameterized.named_parameters(
      ("invalid_lhs_int4", jnp.int4, jnp.int4, "quant_dtype must be"),
      ("int8_with_int4", jnp.int8, jnp.int4, "rhs_quant_dtype must match"),
      (
          "fp8_with_int8",
          jnp.float8_e4m3fn,
          jnp.int8,
          "rhs_quant_dtype must match",
      ),
  )
  def test_validate_quant_dtype_pair_invalid(
      self, quant_dt, rhs_quant_dt, err_regex
  ):
    """Verifies unsupported quant_dtype pairs raise ValueError."""
    with self.assertRaisesRegex(ValueError, err_regex):
      quantized_matmul.validate_quant_dtype_pair(quant_dt, rhs_quant_dt)

  @parameterized.named_parameters(
      (
          "int32_lhs",
          jnp.int32,
          jnp.int32,
          "float8_e4m3fn",
          None,
          None,
          "lhs.dtype must be bfloat16 or float32",
      ),
      (
          "mismatched_rhs_dtype",
          jnp.bfloat16,
          jnp.float32,
          "float8_e4m3fn",
          None,
          None,
          "rhs.dtype .* must match lhs.dtype",
      ),
      (
          "invalid_lhs_quant_dtype_int4",
          jnp.bfloat16,
          jnp.bfloat16,
          "int4",
          None,
          None,
          "quant_dtype must be float8_e4m3fn, float8_e5m2, or int8",
      ),
      (
          "invalid_lhs_quant_dtype_uint8",
          jnp.bfloat16,
          jnp.bfloat16,
          "uint8",
          None,
          None,
          "quant_dtype must be float8_e4m3fn, float8_e5m2, or int8",
      ),
      (
          "invalid_rhs_quant_pair_int8_int4",
          jnp.bfloat16,
          jnp.bfloat16,
          "int8",
          "int4",
          None,
          "rhs_quant_dtype must match quant_dtype or be 'int4'",
      ),
      (
          "invalid_rhs_quant_pair_fp8_int8",
          jnp.bfloat16,
          jnp.bfloat16,
          "float8_e4m3fn",
          "int8",
          None,
          "rhs_quant_dtype must match quant_dtype or be 'int4'",
      ),
      (
          "invalid_out_dtype",
          jnp.bfloat16,
          jnp.bfloat16,
          "float8_e4m3fn",
          None,
          jnp.int32,
          "out_dtype must be bfloat16 or float32",
      ),
  )
  def test_quantized_matmul_rejects_invalid_dtypes(
      self, lhs_dt, rhs_dt, quant_dt, rhs_quant_dt, out_dt, err_regex
  ):
    """Verifies invalid operand, quant, and output dtypes raise ValueError."""
    lhs = jnp.ones((64, 256), dtype=lhs_dt)
    rhs = jnp.ones((256, 128), dtype=rhs_dt)
    with self.assertRaisesRegex(ValueError, err_regex):
      quantized_matmul.quantized_matmul(
          lhs,
          rhs,
          sc_size=128,
          block_m=64,
          block_k=256,
          block_n=128,
          quant_dtype=quant_dt,
          rhs_quant_dtype=rhs_quant_dt,
          out_dtype=out_dt,
          interpret=True,
      )

  @parameterized.named_parameters(
      ("non_2d_lhs", (256,), (256, 128), "must be 2D"),
      ("non_2d_rhs", (64, 256), (1, 256, 128), "must be 2D"),
      (
          "k_mismatch",
          (64, 256),
          (128, 128),
          "Contracting dimension mismatch",
      ),
  )
  def test_quantized_matmul_rejects_invalid_shapes(
      self, lhs_shape, rhs_shape, err_regex
  ):
    """Verifies non-2D inputs and K mismatches raise ValueError."""
    lhs = jnp.ones(lhs_shape, dtype=jnp.bfloat16)
    rhs = jnp.ones(rhs_shape, dtype=jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, err_regex):
      quantized_matmul.quantized_matmul(
          lhs,
          rhs,
          sc_size=128,
          block_m=64,
          block_k=256,
          block_n=128,
          interpret=True,
      )


if __name__ == "__main__":
  absltest.main()
