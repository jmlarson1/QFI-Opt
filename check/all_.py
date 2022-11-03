#!/usr/bin/env python3
import sys
import textwrap

import check_utils
import flake8_
import format_
import mypy_


def run(*args: str) -> int:

    parser = check_utils.get_file_parser()
    parser.description = textwrap.dedent(
        """
        Runs all checks on the repository.
        Exits immediately upon any failure unless passed -f or --force as an argument.
        This script exits with a succeeding exit code if and only if all checks pass.
        """
    )

    parser.add_argument("-f", "--force", action="store_true", help="Continue past (i.e., do not exit after) failing checks.")

    parsed_args = parser.parse_args(args)
    exit_on_failure = not parsed_args.force

    checks_failed = 0
    checks_failed |= format_.run(*parsed_args.files, exit_on_failure=exit_on_failure)
    checks_failed |= flake8_.run(*parsed_args.files, exit_on_failure=exit_on_failure)
    checks_failed |= mypy_.run(*parsed_args.files, exit_on_failure=exit_on_failure)

    return checks_failed


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
