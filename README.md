![image info](./images/qfi-opt.png)

QFI-Opt collects custom simulation packages and configurations for numerical
optimization method that exploit problem structures in 
[quantum fisher information (QFI)](https://en.wikipedia.org/wiki/Quantum_Fisher_information) 
optimization tasks. This repository provides a Python-based
framework that interfaces seamlessly with simulation evaluation scripts,
allowing researchers to efficiently tackle complex optimization problems in
quantum metrology. This package includes robust methods for defining objective
functions, incorporating auxiliary information, and building or updating models
dynamically, ensuring flexibility and scalability for various quantum
optimization scenarios.

See the associated publications arxiv:2406.01859 and arxiv.org:2311.17275 for a
deeper understanding of the underlying principles and the wide range of
applications enabled by QFI-Opt.

## Installation

This package requires Python>=3.10, and can be installed from PyPI with
```
pip install qfi-opt
```
To install from source:
```
git clone https://github.com/jmlarson1/QFI-Opt.git
pip install -e QFI-Opt
```
