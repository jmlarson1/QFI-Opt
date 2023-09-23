#!/usr/bin/env python3
import sys

import general_superstaq.check

if __name__ == "__main__":
    skip_args = ["--skip", "coverage", "requirements", "build_docs", "--"]
    exclude_args = ["--exclude", "qfi_opt/examples/*"]
    exit(general_superstaq.check.all_.run(*skip_args, *exclude_args, *sys.argv[1:]))
