"""TPU v6e (Trillium) hardware specification."""

from accelerator_microbenchmarks.specs import schema

# See also spec in https://docs.cloud.google.com/tpu/docs/v6e
V6E_HARDWARE_SPEC = schema.HardwareSpec(
    name=schema.TpuVersion.V6E,
    topology_dimension=2,
    devices_per_chip=1,
    tflops=schema.TflopsSpec(
        peak_tflops_per_device={
            "bfloat16": 918.0,
            "float32": 459.0,
            "float8_e5m2": 918.0,
            "float8_e4m3fn": 918.0,
            "int8": 1836.0,
            "int4": 3672.0,
        }
    ),
    ici=schema.IciSpec(
        peak_bw_gbps=800.0,
        bidirectional=True,
    ),
    hbm=schema.HbmSpec(peak_bw_gbps=1638.4),
)
