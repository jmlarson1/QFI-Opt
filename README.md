# qfi-opt

This repository contains codes for optimizing the [quantum fisher information (QFI)](https://en.wikipedia.org/wiki/Quantum_Fisher_information) of quantum systems.

Python library dependencies for this repository specified in `requirements.txt`, and can be installed with `pip install -r requirements.txt`.  These requirements are currently rather strict, but will be relaxed eventually.

There are four test scripts that check python codes in this repository:
- `check/format_.py` tests adherence to `black` and `isort` formatting guidelines.  If this test fails, you can run `check/format_.py --apply` to apply the corresponding fixes.
- `check/flake8_.py` tests adherence to `flake8` formatting guidelines.  These must be fixed manually.
- `check/mypy_.py` is a typechecker.
- `check/all_.py` runs all of the above tests in order.
