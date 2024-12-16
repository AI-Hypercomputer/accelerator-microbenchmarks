"""Benchmarks various convolution operations.

Exploring different kernel sizes, strides, padding strategies, and input/output
channel configurations. Including multiple convolution types, such as 2D, 3D,
and depthwise convolutions.
"""

# pylint: disable=g-importing-member
from functools import partial
from typing import Any, Callable, Dict, Tuple

from benchmark_utils import simple_timeit
import jax
import jax.numpy as jnp
import numpy as np
# pylint: disable=g-importing-member


def convolve_common(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    padding_mode: str,
    convolve_fn: Callable[..., Any],
    task_name: str,
) -> Dict[str, Any]:
  """A helper function to run the convolution benchmark.

  Args:
      input_shape (tuple): Shape of the input tensor.
      kernel_shape (tuple): Shape of the convolution kernel.
      padding_mode (str): Padding mode ('same', 'valid', 'full').
      convolve_fn (function): Convolution function to be used.
      task_name (str): The name of the benchmark task for logging.

  Returns:
      A dictionary containing the average time taken for the convolution
      operation and the output shape.
  """

  @partial(jax.jit, static_argnames=["mode"])
  def convolve(x, kernel, mode):
    return convolve_fn(x, kernel, mode=mode)

  x = jnp.arange(np.prod(input_shape)).reshape(input_shape).astype(jnp.bfloat16)
  kernel = (
      jnp.arange(np.prod(kernel_shape))
      .reshape(kernel_shape)
      .astype(jnp.bfloat16)
  )

  # Warm up
  output = convolve(x, kernel, padding_mode).block_until_ready()

  print(f"{task_name} Benchmark:")
  print(
      f"Input Shape: {input_shape}, Kernel Shape: {kernel_shape}, Output Shape:"
      f" {output.shape}, Padding Mode: {padding_mode}"
  )

  # Time the operation
  average_time_ms = simple_timeit(
      convolve, x, kernel, padding_mode, task=task_name
  )
  return {"average_time_ms": average_time_ms, "output_shape": output.shape}


def convolve_common_calculate_metrics(
    # pylint: disable=unused-argument
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    padding_mode: str,
    average_time_ms: float,
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
  """Helper function to calculate the metrics for the convolution benchmarks."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metrics_keys = {"average_time_ms"}
  metadata = {
      key: value
      for key, value in params
      if value is not None and key not in metrics_keys
  }
  metrics = {
      key: value
      for key, value in params
      if value is not None and key in metrics_keys
  }

  kernel_size = np.prod(
      np.array(kernel_shape, dtype=np.int64)
  )  # Total elements in the kernel
  output_size = np.prod(
      np.array(output_shape, dtype=np.int64)
  )  # Total elements in the output
  # Calculate FLOPs (2 * output_size * kernel_size for each output element)
  flops = 2 * output_size * kernel_size

  # Calculate FLOPS utilization
  flops_per_sec = flops / (average_time_ms / 1000)  # Convert ms to seconds

  # Print results
  print(f"Total flops: {flops}")
  print(f"Average Execution Time: {average_time_ms:.4f} ms")
  print(f"FLOPS Utilization: {flops_per_sec / 1e9:.2f} GFLOPS/sec\n")

  # Gather the metrics to report.
  metrics.update({
      "gflops_per_sec": flops_per_sec / 1e9,
  })
  metrics = {key: value for key, value in metrics.items() if value is not None}
  return metadata, metrics


def numpy_convolve(
    input_size: int,
    kernel_size: int,
    padding_mode: str = "same",
) -> float:
  """Benchmarks 1D convolution with jax.numpy.convolve."""

  def convolve_fn(x, kernel, mode):
    return jnp.convolve(x, kernel, mode=mode)

  input_shape = (input_size,)
  kernel_shape = (kernel_size,)
  return convolve_common(
      input_shape, kernel_shape, padding_mode, convolve_fn, "numpy_convolve"
  )


def numpy_convolve_calculate_metrics(
    input_size: int,
    kernel_size: int,
    padding_mode: str,
    output_shape: Tuple[int, ...],
    average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the numpy_convolve benchmark."""
  input_shape = (input_size,)
  kernel_shape = (kernel_size,)
  return convolve_common_calculate_metrics(
      input_shape=input_shape,
      kernel_shape=kernel_shape,
      padding_mode=padding_mode,
      output_shape=output_shape,
      average_time_ms=average_time_ms,
  )


def scipy_signal_convolve(
    input_size: int,
    kernel_size: int,
    dimension: int,
    padding_mode: str = "same",
) -> float:
  """Benchmarks N-dimensional convolution using jax.scipy.signal.convolve."""

  def convolve_fn(x, kernel, mode):
    return jax.scipy.signal.convolve(x, kernel, mode=mode)

  input_shape = tuple([input_size] * dimension)
  kernel_shape = tuple([kernel_size] * dimension)
  return convolve_common(
      input_shape,
      kernel_shape,
      padding_mode,
      convolve_fn,
      "scipy_signal_convolve",
  )


def scipy_signal_convolve_calculate_metrics(
    input_size: int,
    kernel_size: int,
    dimension: int,
    padding_mode: str,
    output_shape: Tuple[int, ...],
    average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the scipy_signal_convolve benchmark."""
  input_shape = tuple([input_size] * dimension)
  kernel_shape = tuple([kernel_size] * dimension)
  return convolve_common_calculate_metrics(
      input_shape=input_shape,
      kernel_shape=kernel_shape,
      padding_mode=padding_mode,
      output_shape=output_shape,
      average_time_ms=average_time_ms,
  )


def scipy_signal_convolve2d(
    input_size: int,
    kernel_size: int,
    padding_mode: str = "same",
) -> float:
  """Benchmarks 2D convolution using jax.scipy.signal.convolve2d."""
  input_shape = (input_size, input_size)
  kernel_shape = (kernel_size, kernel_size)

  def convolve_fn(x, kernel, mode):
    return jax.scipy.signal.convolve2d(x, kernel, mode=mode)

  return convolve_common(
      input_shape,
      kernel_shape,
      padding_mode,
      convolve_fn,
      "scipy_signal_convolve2d",
  )


def scipy_signal_convolve2d_calculate_metrics(
    input_size: int,
    kernel_size: int,
    padding_mode: str,
    output_shape: Tuple[int, ...],
    average_time_ms: float,
) -> Dict[str, Any]:
  """Calculates the metrics for the scipy_signal_convolve2d benchmark."""
  input_shape = (input_size, input_size)
  kernel_shape = (kernel_size, kernel_size)
  return convolve_common_calculate_metrics(
      input_shape=input_shape,
      kernel_shape=kernel_shape,
      padding_mode=padding_mode,
      output_shape=output_shape,
      average_time_ms=average_time_ms,
  )


def lax_conv_general_dilated(
    batch_size: int,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    in_channel: int,
    out_channel: int,
    padding_mode: str,
    stride: int,
    dilation: int,
    dtype: jax.numpy.dtype,
    dimension_numbers: Tuple[str, str, str] = ("NHWC", "HWIO", "NHWC"),
) -> float:
  """Benchmarks convolution with jax.lax.conv_general_dilated."""

  input_shape = (batch_size, input_h, input_w, in_channel)
  kernel_shape = (kernel_h, kernel_w, in_channel, out_channel)
  stride = (stride, stride)
  dilation = (dilation, dilation)

  x = jnp.arange(np.prod(input_shape)).reshape(input_shape).astype(dtype)
  kernel = jnp.arange(np.prod(kernel_shape)).reshape(kernel_shape).astype(dtype)

  @partial(jax.jit, static_argnames=["mode", "stride", "dilation"])
  def convolve(x, kernel, stride, dilation, mode):
    return jax.lax.conv_general_dilated(
        x,
        kernel,
        window_strides=stride,
        padding=mode.upper(),
        rhs_dilation=dilation,
        dimension_numbers=dimension_numbers,
    )

  # Run once.
  output = convolve(
      x, kernel, stride, dilation, padding_mode
  ).block_until_ready()

  print("lax_conv_general_dilated Benchmark:")
  print(
      f"Input Shape: {input_shape}, Kernel Shape: {kernel_shape}, Output shape:"
      f" {output.shape} Stride: {stride}, Dilation: {dilation}, Padding Mode:"
      f" {padding_mode}"
  )

  # Time the operation
  average_time_ms = simple_timeit(
      convolve,
      x,
      kernel,
      stride,
      dilation,
      padding_mode,
      task="lax_conv_general_dilated",
  )
  return {"average_time_ms": average_time_ms, "output_shape": output.shape}


def lax_conv_general_dilated_calculate_metrics(
    # pylint: disable=unused-argument
    batch_size: int,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    in_channel: int,
    out_channel: int,
    padding_mode: str,
    stride: int,
    dilation: int,
    dtype: jax.numpy.dtype,
    output_shape: Tuple[int, ...],
    average_time_ms: float,
    # pylint: disable=unused-argument
) -> Dict[str, Any]:
  """Calculates the metrics for the lax_conv_general_dilated benchmark."""
  # Build dictionary of all the parameters in the function
  params = locals().items()
  metrics_keys = {"average_time_ms"}
  metadata = {
      key: value
      for key, value in params
      if value is not None and key not in metrics_keys
  }
  metrics = {
      key: value
      for key, value in params
      if value is not None and key in metrics_keys
  }

  # Number of output elements
  output_size = np.prod(
      np.array(output_shape, dtype=np.int64)
  )  # Total elements in the output

  # Operations per output element:
  # Each output element requires kernel_h * kernel_w * in_channels
  # multiply-accumulates
  ops_per_element = kernel_h * kernel_w * in_channel

  # Total FLOPs = output_elements * operations_per_element * 2
  # Multiply by 2 because each multiply-accumulate (MAC) = 1 multiply + 1 add
  # = 2 FLOPs
  flops = output_size * ops_per_element * 2

  # Calculate FLOPS utilization
  flops_per_sec = flops / (average_time_ms / 1000)  # Convert ms to seconds

  # Print results
  print(f"Total flops: {flops}")
  print(f"Average Execution Time: {average_time_ms:.4f} ms")
  print(f"FLOPS Utilization: {flops_per_sec / 1e9:.2f} GFLOPS/sec\n")
  # Gather the metrics to report.
  metrics.update({
      "gflops_per_sec": flops_per_sec / 1e9,
  })
  return metadata, metrics
