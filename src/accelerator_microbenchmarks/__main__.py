"""Module execution entry point for accelerator_microbenchmarks."""

import sys
from accelerator_microbenchmarks import cli

if __name__ == "__main__":
  cli.main(sys.argv[1:])
