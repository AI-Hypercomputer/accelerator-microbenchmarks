import os
import yaml

# Constants
COMMON_MATRIX_DIM_RANGE = {"start": 2, "end": 131072, "multiplier": 2}
COMMON_DTYPE = "float32"
COMMON_NUM_RUNS = 5


class TopologyConfig:

  def __init__(self, name, x, y, z, mesh_shapes_1d, sharding_1d):
    self.name = name
    self.x = x
    self.y = y
    self.z = z
    # Assuming 2 cores per chip (e.g. 4x4x4 chips = 64 chips = 128 cores)
    self.ici_size = x * y * z * 2
    self.mesh_shapes_1d = mesh_shapes_1d
    self.sharding_1d = sharding_1d

  def get_1d_params(self):
    params = []
    # Parallel Replica (Index 0)
    params.append(
        self._create_param(self.mesh_shapes_1d[0], self.sharding_1d[0], 1)
    )
    # Non-Parallel Replica Z, Y, X (Indices 1, 2, 3)
    for i in range(1, 2):
      params.append(
          self._create_param(self.mesh_shapes_1d[i], self.sharding_1d[i], 1)
      )
    return params

  def get_2d_params(self):
    # Item 1: x by (y*z) by 2
    mesh1 = f"{self.x}x{self.y * self.z}x2"
    shard1 = f"1x{self.y * self.z}x1"

    # Item 2: x by y by (z*2)
    mesh2 = f"{self.x}x{self.y}x{self.z * 2}"
    shard2 = f"1x{self.y}x{self.z * 2}"

    return [
        self._create_param(mesh1, shard1, 2),
        self._create_param(mesh2, shard2, 2),
    ]

  def get_3d_params(self):
    # Item 1: (x*y*z) by 2
    mesh1 = f"{self.x * self.y * self.z}x2"
    shard1 = f"{self.x * self.y * self.z}x1"

    # Item 2: x by y by (z*2)
    mesh2 = f"{self.x}x{self.y}x{self.z * 2}"
    shard2 = f"{self.x}x{self.y}x{self.z * 2}"

    return [
        self._create_param(mesh1, shard1, 3),
        self._create_param(mesh2, shard2, 3),
    ]

  def _create_param(self, mesh_shape, sharding_strategy, op_dimension):
    return FlowMap({
        "matrix_dim_range": COMMON_MATRIX_DIM_RANGE,
        "dtype": COMMON_DTYPE,
        "mesh_shape": mesh_shape,
        "ici_size_range": self.ici_size,
        "sharding_strategy": sharding_strategy,
        "op_dimension": op_dimension,
        "num_runs": COMMON_NUM_RUNS,
    })


TOPOLOGIES = [
    TopologyConfig(
        "4x4x4",
        4,
        4,
        4,
        mesh_shapes_1d=["16x4x2", "4x4x8"],
        sharding_1d=["1x4x1", "1x1x8"],
    ),
    TopologyConfig(
        "8x8x8",
        8,
        8,
        8,
        # Corrected 32x8x2 -> 64x8x2 to match ici_size 1024
        mesh_shapes_1d=["64x8x2", "8x8x16"],
        sharding_1d=["1x8x1", "1x1x16"],
    ),
    TopologyConfig(
        "8x8x16",
        8,
        8,
        16,
        mesh_shapes_1d=["64x16x2", "8x8x32"],
        sharding_1d=["1x16x1", "1x1x32"],
    ),
]


def create_config_content(benchmark_name, suffix, params):
  return {
      "benchmarks": [{
          "benchmark_name": benchmark_name,
          "benchmark_sweep_params": params,
          "trace_dir": f"../microbenchmarks/{benchmark_name}{suffix}",
          "csv_path": f"../microbenchmarks/{benchmark_name}{suffix}",
          "xlml_metrics_dir": f"../microbenchmarks/{benchmark_name}{suffix}",
          "xla_dump_dir": (
              f"../microbenchmarks/{benchmark_name}{suffix}/hlo_graphs"
          ),
      }]
  }


# Disable YAML aliases and support FlowStyle for params
class FlowMap(dict):
  """Custom dict to render as flow-style (inline) YAML."""

  pass


class CustomDumper(yaml.SafeDumper):

  def ignore_aliases(self, data):
    return True


def flow_map_representer(dumper, data):
  return dumper.represent_mapping(
      "tag:yaml.org,2002:map", data, flow_style=True
  )


CustomDumper.add_representer(FlowMap, flow_map_representer)


def main():
  # Definition of all configs to generate
  # Format: (file_prefix, benchmark_name_in_config, params_getter, file_suffix_yaml, dir_suffix)
  # The dir_suffix is used for trace_dir etc. e.g. "psum_1d"

  config_defs = [
      # all_gather
      (
          "all_gather",
          "all_gather",
          lambda t: t.get_1d_params(),
          "_1d.yaml",
          "_1d",
      ),
      (
          "all_gather",
          "all_gather",
          lambda t: t.get_2d_params(),
          "_2d.yaml",
          "_2d",
      ),
      (
          "all_gather",
          "all_gather",
          lambda t: t.get_3d_params(),
          "_3d.yaml",
          "_3d",
      ),
      # all_reduce (psum)
      ("all_reduce", "psum", lambda t: t.get_1d_params(), "_1d.yaml", "_1d"),
      ("all_reduce", "psum", lambda t: t.get_2d_params(), "_2d.yaml", "_2d"),
      ("all_reduce", "psum", lambda t: t.get_3d_params(), "_3d.yaml", "_3d"),
      # reduce_scatter (psum_scatter)
      (
          "reduce_scatter",
          "psum_scatter",
          lambda t: t.get_1d_params(),
          "_1d.yaml",
          "_1d",
      ),
      (
          "reduce_scatter",
          "psum_scatter",
          lambda t: t.get_2d_params(),
          "_2d.yaml",
          "_2d",
      ),
      (
          "reduce_scatter",
          "psum_scatter",
          lambda t: t.get_3d_params(),
          "_3d.yaml",
          "_3d",
      ),
      # all_to_all (subset of params)
      (
          "all_to_all",
          "all_to_all",
          lambda t: t.get_1d_params()[1:4],
          "_1d.yaml",
          "_1d",
      ),
      (
          "all_to_all",
          "all_to_all",
          lambda t: [t.get_2d_params()[1]],
          "_2d.yaml",
          "_2d",
      ),
      (
          "all_to_all",
          "all_to_all",
          lambda t: [t.get_3d_params()[1]],
          "_3d.yaml",
          "_3d",
      ),
  ]

  for topo in TOPOLOGIES:
    os.makedirs(topo.name, exist_ok=True)
    for prefix, bench_name, params_fn, yaml_suffix, dir_suffix in config_defs:
      params = params_fn(topo)
      config = create_config_content(bench_name, dir_suffix, params)

      filename = f"{prefix}{yaml_suffix}"
      filepath = os.path.join(topo.name, filename)

      with open(filepath, "w") as f:
        yaml.dump(config, f, Dumper=CustomDumper, sort_keys=False, width=2048)

  print("Config files generated successfully.")


if __name__ == "__main__":
  main()
