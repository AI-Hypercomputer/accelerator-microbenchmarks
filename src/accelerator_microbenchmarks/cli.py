"""Canonical CLI entry point for the TPU Microbenchmark Suite (TPUMS)."""

import json
import os
import sys
from typing import List, Optional, Sequence

import simple_parsing
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import platform as core_platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import runner


def _add_common_execution_args(parser) -> None:
  """Adds common output, profiling, and hardware flags to a subparser."""
  parser.add_argument(
      "--output_dir",
      type=str,
      default="results",
      help="Directory to save persistent benchmark reports.",
  )
  parser.add_argument(
      "--profile_dir",
      type=str,
      default="/tmp/tensorboard",
      help="Directory to collect and save profiling trace files.",
  )
  parser.add_argument(
      "--hw",
      type=str,
      default=None,
      help="Hardware target environment (e.g. ironwood, gfc).",
  )


def create_parser() -> simple_parsing.ArgumentParser:
  """Constructs the hierarchical resource-action CLI parser for tpums."""
  parser = simple_parsing.ArgumentParser(
      prog="tpums",
      description=(
          "TPU Microbenchmark Suite (TPUMS) CLI tool for performance"
          " evaluation on TPU Platforms."
      ),
  )

  # Level 1: Resources (platform, benchmark)
  resource_subparsers = parser.add_subparsers(
      dest="resource", required=True, title="Commands"
  )

  # -------------------------------------------------------------------------
  # 1. Resource: platform
  # -------------------------------------------------------------------------
  platform_parser = resource_subparsers.add_parser(
      "platform", help="Query and display TPU hardware topology and metadata."
  )
  platform_action_subparsers = platform_parser.add_subparsers(
      dest="action", required=True, title="Platform Actions"
  )
  platform_action_subparsers.add_parser(
      "describe", help="Query and display TPU hardware topology and metadata."
  )

  # -------------------------------------------------------------------------
  # 2. Resource: benchmark
  # -------------------------------------------------------------------------
  benchmark_parser = resource_subparsers.add_parser(
      "benchmark", help="Benchmark execution and task discovery."
  )
  benchmark_action_subparsers = benchmark_parser.add_subparsers(
      dest="action", required=True, title="Benchmark Actions"
  )

  # Action: benchmark list
  benchmark_action_subparsers.add_parser(
      "list", help="List all registered microbenchmark tasks."
  )

  # Action: benchmark run-config <path_to_yaml>
  run_config_parser = benchmark_action_subparsers.add_parser(
      "run-config",
      help=(
          "Execute a benchmark suite defined in a YAML configuration file."
      ),
  )
  run_config_parser.add_argument(
      "config_path", type=str, help="Path to the YAML configuration file."
  )
  _add_common_execution_args(run_config_parser)

  # Action: benchmark run <task> [options]
  run_parser = benchmark_action_subparsers.add_parser(
      "run",
      help="Execute a benchmark task interactively via CLI flags.",
  )
  task_subparsers = run_parser.add_subparsers(
      dest="task", required=True, title="Supported Tasks"
  )

  # Dynamically register all non-experimental benchmark tasks and their Config dataclasses
  benchmark_loader.load_all_benchmarks()
  for task_name in registry.benchmark_registry.list_benchmark_names(
      include_experimental=False, include_aliases=False
  ):
    bench_cls = registry.benchmark_registry.get_benchmark(task_name)
    task_parser = task_subparsers.add_parser(
        task_name, help=f"Run {task_name} benchmark."
    )
    _add_common_execution_args(task_parser)
    task_parser.add_arguments(bench_cls.Config, dest="task_config")

  return parser


def run(argv: Sequence[str]) -> None:
  """Application entry point for parsing and executing tpums commands.

  Parses application-level arguments (subcommands, YAML configs, benchmarks)
  using simple_parsing. Framework-agnostic and directly testable.
  """
  if not argv:
    create_parser().print_help()
    return

  parser = create_parser()
  args, _ = parser.parse_known_args(argv)

  # 1. Handle `tpums platform describe`
  if args.resource == "platform" and args.action == "describe":
    try:
      desc = core_platform.get_platform_description()
      if desc.get("tpu_type") == "none":
        print(
            "WARNING: Running in non-TPU (CPU) environment. No TPU devices"
            " detected.",
            file=sys.stderr,
        )
      print(json.dumps(desc, indent=2))
      return
    except RuntimeError as e:
      print(f"Error: {e}", file=sys.stderr)
      sys.exit(1)

  # 2. Handle `tpums benchmark list`
  if args.resource == "benchmark" and args.action == "list":
    tasks = registry.benchmark_registry.list_benchmark_names(
        include_experimental=False, include_aliases=False
    )
    output = [{"task": task} for task in tasks]
    print(json.dumps(output, indent=2))
    return

  # 3. Handle `tpums benchmark run-config <path_to_yaml>`
  if args.resource == "benchmark" and args.action == "run-config":
    resolved_path = args.config_path
    if not os.path.exists(resolved_path) and os.path.exists(args.config_path):
      resolved_path = args.config_path

    raw_configs = config.load_config(resolved_path)
    tasks = []
    for raw in raw_configs:
      name = raw.pop("name")
      bench_cls = registry.benchmark_registry.get_benchmark(name)
      tasks.append((name, bench_cls.Config(**raw)))

    runner.run_benchmarks(
        tasks=tasks,
        output_dir=args.output_dir,
        hw=args.hw,
        xprof_dir=args.profile_dir,
        config_path=args.config_path,
    )
    return

  # 4. Handle `tpums benchmark run <task> [options]`
  if args.resource == "benchmark" and args.action == "run":
    runner.run_benchmarks(
        tasks=[(args.task, args.task_config)],
        output_dir=args.output_dir,
        hw=args.hw,
        xprof_dir=args.profile_dir,
    )
    return


def main() -> None:
  """Top-level executable entry point."""
  # Open Source: Standard Python entry point.
  run(sys.argv[1:])


if __name__ == "__main__":
  main()
