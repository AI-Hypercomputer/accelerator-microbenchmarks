"""Canonical CLI entry point for the TPU Microbenchmark Suite (TPUMS)."""

import argparse
import dataclasses
import enum
import json
import os
import sys
from typing import Any, Sequence, Type

from absl import app
from absl import flags
from accelerator_microbenchmarks.benchmarks import benchmark_loader
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import config
from accelerator_microbenchmarks.core import platform as core_platform
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import runner


def _add_common_execution_args(parser: argparse.ArgumentParser) -> None:
  """Adds common output and profiling flags to a subparser."""
  parser.add_argument(
      "--output_dir",
      type=str,
      default="results",
      help="Directory to save persistent benchmark reports.",
  )
  parser.add_argument(
      "--xprof_dir",
      type=str,
      default="/tmp/tensorboard",
      help="Directory to collect and save profiling trace files.",
  )
  parser.add_argument(
      "--xla_flags_file_path",
      type=str,
      default=None,
      help="Optional path to custom op_flags YAML file.",
  )


def _str_to_bool(value: str) -> bool:
  """Converts a CLI string token into a boolean."""
  lowered = value.lower()
  if lowered == "true":
    return True
  if lowered == "false":
    return False
  raise argparse.ArgumentTypeError(
      f"invalid boolean value: '{value}' (choose from 'true', 'false')"
  )


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    config_cls: Type[base.BaseBenchmarkParams],
) -> None:
  """Registers dataclass fields as argparse flags.

  Flags are grouped under a `<ConfigClass> parameters` section. Boolean, enum,
  numeric, and string flags all take explicit values and support `nargs="+"`;
  booleans accept the literal tokens `true` and `false`.

  Args:
    parser: The (sub)parser to register the flags on.
    config_cls: The benchmark configuration dataclass (e.g. `AllReduceParams`).
  """
  parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
  group = parser.add_argument_group(
      title=f"{config_cls.__name__} parameters",
  )

  for field in dataclasses.fields(config_cls):
    if not field.init:
      continue

    flag_names = [f"--{field.name}"]
    # Preserve the single-dash shorthand (e.g. `-m`) for one-letter fields.
    if len(field.name) == 1:
      flag_names.append(f"-{field.name}")

    default = (
        field.default if field.default is not dataclasses.MISSING else None
    )
    help_text = field.metadata.get("help", "").strip()

    # 1. Boolean flags: `--transpose_a true` or `--transpose_a true false`
    if field.type is bool:
      group.add_argument(
          *flag_names,
          type=_str_to_bool,
          nargs="+",
          default=default,
          metavar="{true,false}",
          help=help_text,
      )
    # 2. Enum flags (strictly lowercase values)
    elif isinstance(field.type, type) and issubclass(field.type, enum.Enum):
      default_val = (
          default.value if isinstance(default, enum.Enum) else default
      )
      group.add_argument(
          *flag_names,
          type=str,
          nargs="+",
          choices=[e.value for e in field.type],
          default=default_val,
          help=help_text,
      )
    # 3. Numeric / string flags
    else:
      # Non-class annotations (e.g. `Optional[str]`) fall back to str parsing;
      # the dataclass itself remains the source of truth for validation.
      field_type = field.type if isinstance(field.type, type) else str
      group.add_argument(
          *flag_names,
          type=field_type,
          nargs="+",
          default=default,
          help=help_text,
      )


def parse_cli_config(
    config_cls: Type[base.BaseBenchmarkParams],
    parsed_args: argparse.Namespace,
) -> base.BaseBenchmarkParams:
  """Builds a BaseBenchmarkParams instance from parsed CLI arguments.

  Args:
    config_cls: The benchmark configuration dataclass to instantiate.
    parsed_args: Namespace produced by a parser configured via
      `add_dataclass_arguments`.

  Returns:
    An instance of `config_cls`.
  """
  kwargs: dict[str, Any] = {}
  for field in dataclasses.fields(config_cls):
    if not field.init or not hasattr(parsed_args, field.name):
      continue
    value = getattr(parsed_args, field.name)
    ## Does not support multiple values for a single flag.
    kwargs[field.name] = value[0] if isinstance(value, list) else value
  return config_cls(**kwargs)


def create_parser() -> argparse.ArgumentParser:
  """Constructs the hierarchical resource-action CLI parser for tpums."""
  parser = argparse.ArgumentParser(
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

  # Dynamically register all non-experimental benchmark tasks and their Config
  # dataclasses
  benchmark_loader.load_all_benchmarks()
  for task_name in registry.benchmark_registry.list_benchmark_names(
      include_experimental=False, include_aliases=False
  ):
    bench_cls = registry.benchmark_registry.get_benchmark(task_name)
    task_parser = task_subparsers.add_parser(
        task_name,
        help=f"Run {task_name} benchmark.",
    )
    _add_common_execution_args(task_parser)
    task_parser.add_argument(
        "--xprof_timing",
        action="store_true",
        default=False,
        help="Enable XProf trace collection and device timing analysis.",
    )
    add_dataclass_arguments(task_parser, bench_cls.Config)

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
      desc = core_platform.get_platform_info()
      print(json.dumps(dataclasses.asdict(desc), indent=2))
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
    effective_xprof_timing = (
        bool(raw_configs[0].get("xprof_timing", False))
        if raw_configs
        else False
    )

    tasks = []
    for raw in raw_configs:
      name = raw.pop("name")
      raw.pop("xprof_timing", None)
      bench_cls = registry.benchmark_registry.get_benchmark(name)
      tasks.append((name, bench_cls.Config(**raw)))

    runner.run_benchmarks(
        tasks=tasks,
        output_dir=args.output_dir,
        xprof_timing=effective_xprof_timing,
        xprof_dir=args.xprof_dir,
        config_path=args.config_path,
        xla_flags_file_path=args.xla_flags_file_path,
    )
    return

  # 4. Handle `tpums benchmark run <task> [options]`
  if args.resource == "benchmark" and args.action == "run":
    bench_cls = registry.benchmark_registry.get_benchmark(args.task)
    task_config = parse_cli_config(bench_cls.Config, args)
    runner.run_benchmarks(
        tasks=[(args.task, task_config)],
        output_dir=args.output_dir,
        xprof_timing=args.xprof_timing,
        xprof_dir=args.xprof_dir,
        xla_flags_file_path=args.xla_flags_file_path,
    )
    return


def main() -> None:
  """Top-level executable entry point."""
  # Open Source: Standard Python entry point.
  run(sys.argv[1:])


if __name__ == "__main__":
  main()
