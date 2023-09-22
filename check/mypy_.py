#!/usr/bin/env python3
import subprocess
import sys
import textwrap
from typing import Iterable

import check_utils


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = "*ode_jax.py",
) -> int:
    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs mypy on the repository (typing check).
        """
    )

    parsed_args, args_to_pass = parser.parse_known_intermixed_args(args)
    files = parsed_args.files or check_utils.get_tracked_files(include, exclude)

    return subprocess.call(["mypy", *files, *args_to_pass], cwd=check_utils.root_dir)


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
