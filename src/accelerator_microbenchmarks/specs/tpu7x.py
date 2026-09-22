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
            "float32": 576.75,  # 2-pass BF16 emulation on MXU
            "float8_e5m2": 2307.0,
            "float8_e4m3fn": 2307.0,
            "int8": 1153.5,  # Emulated on BF16 datapath
        }
    ),
    ici=schema.IciSpec(
        unidirectional_link_bw_gb_s=1200.0,
        bidirectional=True,
    ),
    hbm=schema.HbmSpec(peak_bw_gb_s=7380.0),
)
