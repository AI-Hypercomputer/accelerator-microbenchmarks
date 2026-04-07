"""Auto-discovery of all benchmark modules."""

import importlib
import pkgutil
import sys


def load_all_benchmarks():
  """Dynamically loads all modules in the benchmarks package to register them."""
  package_name = __name__
  package = sys.modules[package_name]

  # Iterate through all modules in the current package's directory
  for _, module_name, _ in pkgutil.iter_modules(package.__path__):
    full_module_name = f"{package_name}.{module_name}"
    try:
      importlib.import_module(full_module_name)
    except Exception as e:
      print(
          "Warning: Failed to dynamically load benchmark module"
          f" '{full_module_name}': {e}"
      )


load_all_benchmarks()
