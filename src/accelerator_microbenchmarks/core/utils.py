"""Utility functions for accelerator microbenchmarks."""

import jax.numpy as jnp


def parse_dtype(dtype_str: str) -> jnp.dtype:
  """Parses a string into a jax.numpy dtype.

  Args:
    dtype_str: The string representation of the dtype (e.g., 'bfloat16').

  Returns:
    The corresponding jax.numpy dtype.

  Raises:
    ValueError: If the dtype string does not correspond to a valid jnp dtype.
  """
  if not hasattr(jnp, dtype_str):
    raise ValueError(
        f"Invalid dtype string: '{dtype_str}'. Could not find"
        f" jax.numpy.{dtype_str}."
    )
  return getattr(jnp, dtype_str)
