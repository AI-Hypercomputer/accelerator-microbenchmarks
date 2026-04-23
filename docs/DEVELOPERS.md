# Developer Guide: Adding New Benchmarks

This guide explains how to extend `jax-benchmarks` with new operations or
composite benchmarks.

## 1. Benchmark Anatomy

Every benchmark must inherit from `BaseBenchmark` and reside in
`src/jax_benchmarks/benchmarks/`.

```python
from ..core import BaseBenchmark, registry
import jax.numpy as jnp
import jax

@registry.register("my_new_op")
class MyNewOpBenchmark(BaseBenchmark):
    def setup(self, **params):
        """Called once. Use for JIT compilation or constant allocation."""
        @jax.jit
        def my_op(x):
            return x * 2
        self._jit_fn = my_op

    def generate_inputs(self, **params):
        """Generate test data. Returns a tuple of arguments for run_op."""
        size = params.get("size", 1024)
        x = jnp.ones((size,))
        return (x,)

    def run_op(self, x):
        """The core loop operation. Must use jax.block_until_ready internally handled by Base."""
        return self._jit_fn(x)
```

## 2. Advanced: Adding Roofline Support

To enable theoretical performance analysis, implement these two methods:

```python
    def get_total_bytes(self, **params) -> float:
        """Calculate bytes moved to/from HBM."""
        size = params.get("size", 1024)
        return size * 4 * 2 # 1 read + 1 write of float32

    def get_arithmetic_intensity(self, **params) -> float:
        """Flops per Byte moved."""
        size = params.get("size", 1024)
        flops = size # 1 multiply per element
        return flops / self.get_total_bytes(**params)
```

## 3. Registering the Benchmark

Update `src/jax_benchmarks/benchmarks/__init__.py` to import your new module:

```python
def register_all():
    # ... existing imports
    from . import my_new_module
```

## 4. Best Practices

-   **Mesh Awareness**: Use `self.mesh` for any sharding logic to ensure TPU
    multi-core compatibility.
-   **Micro-benchmarks**: Keep kernels focused. Avoid complex state management
    inside `run_op`.
-   **Dtype Flexibility**: Always allow `dtype` or `in_dtype`/`out_dtype` to be
    passed via `params`.
-   **Trace-Ready**: Ensure `run_op` is a pure JAX function to support
    `use_trace_roofline: true`.
-   **Stable Test Names**: Ensure your benchmark has a stable name to enable regression tracking in MLCompass. Avoid adding timestamps to test names.
-   **Numeric Metrics**: Ensure all reported metrics are numeric (int or float) to be compatible with MLCompass.

## 5. Local Verification

Test your new benchmark using a minimal config:

```bash
python3 -m jax_benchmarks.main --config - <<EOF
benchmarks:
  - name: my_new_op
    size: 2048
EOF
```
