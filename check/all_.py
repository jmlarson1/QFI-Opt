#!/usr/bin/env python3
import sys
import textwrap

from general_superstaq.check import check_utils, configs, flake8_, format_, mypy_, pylint_


def run(*args: str) -> int:
    parser = check_utils.get_check_parser()
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
    checks_failed |= configs.run(exit_on_failure=exit_on_failure)
    checks_failed |= format_.run(exit_on_failure=exit_on_failure)
    checks_failed |= flake8_.run(exit_on_failure=exit_on_failure)
    checks_failed |= pylint_.run(exit_on_failure=exit_on_failure)
    checks_failed |= mypy_.run(exit_on_failure=exit_on_failure, exclude="qfi_opt/examples/*")

    return checks_failed


if __name__ == "__main__":
    exit(run(*sys.argv[1:]))
