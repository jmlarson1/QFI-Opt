![image info](./images/qfi-opt.png)

This repository contains codes for optimizing the [quantum fisher information (QFI)](https://en.wikipedia.org/wiki/Quantum_Fisher_information) of quantum systems.

You can install this repository as a local Python package named `qfi_opt` by running `pip install .` in the root directory of this repository.  Run `pip install -e '.[dev]'` to additionally install developer tools and make the installation reflect local changes to the repo.

There are four test scripts that check python codes in this repository:
- `check/format_.py` tests adherence to [`black`](https://black.readthedocs.io/en/stable/) and [`isort`](https://pycqa.github.io/isort/) formatting guidelines.  If this test fails, you can run `check/format_.py --apply` to apply the corresponding fixes.
- `check/flake8_.py` runs the [code linter](https://medium.com/python-pandemonium/what-is-flake8-and-why-we-should-use-it-b89bd78073f2) [`flake8`](https://pypi.org/project/flake8/).
- `check/mypy_.py` runs the typechecker [`mypy`](https://mypy.readthedocs.io/en/stable/).
- `check/all_.py` runs all of the above tests in the order provided.
