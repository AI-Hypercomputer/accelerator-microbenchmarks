"""Utility functions for microbenchmarking."""

import datetime
import os

import jax
import jsonlines


def simple_timeit(f, *args, tries=1000, task=None):
  """Simple utility to time a function for multiple runs."""
  assert task is not None

  jax.block_until_ready(f(*args))  # warm it up!
  s = datetime.datetime.now()
  for _ in range(tries):
    jax.devices()  # Force synchronization across devices
    data = f(*args)
  jax.block_until_ready(data)
  e = datetime.datetime.now()
  average_time_ms = (1000 * (e - s).total_seconds())/tries

  # average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"{task}: average time milliseconds of {tries} tries: {average_time_ms:.2f}")
  return average_time_ms
  # """Simple utility to time a function for multiple runs."""
  # assert task is not None

  # outcomes_ms = []
  # jax.block_until_ready(f(*args))  # warm it up!
  # for _ in range(tries):
  #   jax.devices()  # Force synchronization across devices
  #   s = datetime.datetime.now()
  #   jax.block_until_ready(f(*args))
  #   e = datetime.datetime.now()
  #   outcomes_ms.append(1000 * (e - s).total_seconds())

  # average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  # print(f"{task}: average time milliseconds: {average_time_ms:.2f}")
  # return average_time_ms


def maybe_write_metrics_file(
    metrics_dir, metrics, metadata, test_name, test_start_time, test_end_time
):
  """Writes metrics to a JSONL file to be consumed by the XLML metrics pipeline."""

  # Only write metrics from one host.
  if jax.process_index() != 0:
    return

  jsonl_name = "metrics_report.jsonl"
  jsonl_path = metrics_dir + "/" + jsonl_name
  metadata.update({
      "testsuite": "microbenchmark",
      "test_name": f"{test_name}",
      "test_start_timestamp": f"{test_start_time}",
      "test_end_timestamp": f"{test_end_time}",
  })
  metrics_data = {
      "metrics": metrics,
      "dimensions": metadata,
  }
  # Make sure the metadata value is a string.
  for key, value in metadata.items():
    metadata[key] = str(value)

  # Ensure the directory exists
  os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

  print(f"Writing metrics to JSONL file: {jsonl_path}")
  with jsonlines.open(jsonl_path, mode="a") as writer:
    writer.write(metrics_data)
