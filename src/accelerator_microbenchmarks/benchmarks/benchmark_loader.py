"""Auto-discovery of all benchmark modules."""

import importlib
import os
import pkgutil


def load_all_benchmarks():
  """Dynamically loads all modules in the benchmarks package to register them."""
  package_dir = os.path.dirname(__file__)
  package_name = __package__

  # Iterate through all modules in the current package's directory
  for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
    if is_pkg or module_name == "benchmark_loader":
      continue
    full_module_name = f"{package_name}.{module_name}"
    try:
      importlib.import_module(full_module_name)
    except (ImportError, SyntaxError, RuntimeError) as e:
      print(
          "Warning: Failed to dynamically load benchmark module"
          f" '{full_module_name}': {e}"
      )
