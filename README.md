![image info](./images/qfi-opt.png)

=======
QFI-Opt
=======

This repository contains codes for optimizing the [quantum fisher information (QFI)](https://en.wikipedia.org/wiki/Quantum_Fisher_information) of quantum systems.

Python library dependencies for this repository specified in `requirements.txt`, and can be installed with `pip install -r requirements.txt`.  These requirements are currently rather strict, but will be relaxed eventually.

There are four test scripts that check python codes in this repository:
- `check/format_.py` tests adherence to [`black`](https://black.readthedocs.io/en/stable/) and [`isort`](https://pycqa.github.io/isort/) formatting guidelines.  If this test fails, you can run `check/format_.py --apply` to apply the corresponding fixes.
- `check/flake8_.py` runs the [code linter](https://medium.com/python-pandemonium/what-is-flake8-and-why-we-should-use-it-b89bd78073f2) [`flake8`](https://pypi.org/project/flake8/).
- `check/mypy_.py` runs the typechecker [`mypy`](https://mypy.readthedocs.io/en/stable/).
- `check/all_.py` runs all of the above tests in the order provided.
