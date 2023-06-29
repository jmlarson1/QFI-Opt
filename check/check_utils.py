"""
Dumping ground for check script utilities.
"""
import argparse
import dataclasses
import fnmatch
import os
import subprocess
import sys
from typing import Any, Callable, Iterable, List

# identify the root directory of the "main" script that called this module
main_file_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
root_dir = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=main_file_dir, text=True).strip()


# container for string formatting console codes
@dataclasses.dataclass
class Style:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def styled(text: str, style_code: str) -> str:
    return style_code + text + Style.RESET


def warning(text: str) -> str:
    return styled(text, Style.RED)


def failure(text: str) -> str:
    return styled(text, Style.BOLD + Style.RED)


def success(text: str) -> str:
    return styled(text, Style.BOLD + Style.GREEN)


####################################################################################################
# methods for identifying files to check


def get_tracked_files(
    include: str | Iterable[str],
    exclude: str | Iterable[str] = "",
) -> List[str]:
    """
    Identify all files that (1) are tracked by git in this repo, and (2) match the given include string(s) (interpreted as bash-style regex).
    Optionally excludes anything that matches the given exclude regex.
    """
    match_patterns = [include] if isinstance(include, str) else list(include)
    commands = ["git", "ls-files"] + match_patterns
    matching_files = subprocess.check_output(commands, text=True, cwd=root_dir).splitlines()
    should_include = inclusion_filter(exclude)
    return [file for file in matching_files if should_include(file)]


def inclusion_filter(exclude: str | Iterable[str]) -> Callable[[str], bool]:
    """Construct filter that decides whether a file should be included."""
    if not exclude:
        return lambda _: True

    exclusions = [exclude] if isinstance(exclude, str) else exclude

    def should_include(file: str) -> bool:
        return not any(fnmatch.fnmatch(file, exclusion) for exclusion in exclusions)

    return should_include


def extract_files(
    parsed_args: argparse.Namespace,
    include: str | Iterable[str],
    exclude: str | Iterable[str] = "",
) -> List[str]:
    files = parsed_args.files if "files" in parsed_args else []
    if not files:
        return get_tracked_files(include, exclude)
    else:
        return files


####################################################################################################
# file parsing and a decorator to exit instead of returning a failing exit code


def get_file_parser() -> argparse.ArgumentParser:
    """Build and return a command-line argument parser that accepts file arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    help_text = "The files to check. If not passed any files, inspects the entire repo."
    parser.add_argument("files", nargs="*", help=help_text)
    return parser


def enable_exit_on_failure(func_with_returncode: Callable[..., int]) -> Callable[..., int]:
    """Decorator optionally allowing a function to exit instead of returning a failing return code."""

    def func_with_exit(*args: Any, exit_on_failure: bool = False, **kwargs: Any) -> int:
        returncode = func_with_returncode(*args, **kwargs)
        if exit_on_failure and returncode:
            exit(returncode)
        return returncode

    return func_with_exit
