"""TPU v7x (Ironwood) hardware specification."""

from accelerator_microbenchmarks.specs import schema

# See also spec in https://docs.cloud.google.com/tpu/docs/tpu7x
TPU7X_HARDWARE_SPEC = schema.HardwareSpec(
    name=schema.TpuVersion.TPU7X,
    topology_dimension=3,
    devices_per_chip=2,
    tflops=schema.TflopsSpec(
        peak_tflops_per_device={
            "bfloat16": 1153.5,
            # Same peak as bfloat16: MXU multiplies take bfloat16 inputs and
            # accumulate in FP32, so a float32 matmul at JAX's default
            # precision is one bfloat16 pass (see docs/BENCHMARKS.md).
            # https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
            "float32": 1153.5,
            "float8_e5m2": 2307.0,
            "float8_e4m3fn": 2307.0,
            "int8": 1153.5,  # Emulated on BF16 datapath
            "float4_e2m1fn": None,  # Not natively supported on tpu7x.
        }
    ),
    ici=schema.IciSpec(
        link_bw_per_direction_gb_s=100.0,
        wraparound_rule=schema.WraparoundRule.ALL_DIMS_GE_4,
    ),
    # Host interface: PCIe Gen5 x16 per chip, shared by both chiplets; raw
    # 32 GT/s x 16 lanes / 8.
    pcie=schema.PcieSpec(bandwidth_per_direction_gb_s=64.0),
    hbm=schema.HbmSpec(peak_bw_gb_s=7380.0),
)
