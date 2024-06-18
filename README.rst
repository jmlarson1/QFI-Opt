|qfiopt_logo|

QFI-Opt collects custom simulation packages and configurations for numerical
optimization methods that exploit problem structures in 
`quantum fisher information (QFI) <https://en.wikipedia.org/wiki/Quantum_Fisher_information>`_
optimization tasks. This repository provides a Python-based
framework that interfaces seamlessly with simulation evaluation scripts,
allowing researchers to efficiently tackle complex optimization problems in
quantum metrology. This package includes robust methods for defining objective
functions, incorporating auxiliary information, and building or updating models
dynamically, ensuring flexibility and scalability for various quantum
optimization scenarios.

See the associated publications [arxiv:2406.01859](https://arxiv.org/abs/2406.01859) and [arxiv.org:2311.17275](https://arxiv.org/abs/2311.17275) for a
deeper understanding of the underlying principles and the wide range of
applications enabled by QFI-Opt.

Installation
============

This package requires Python>=3.10, and can be installed from PyPI with::

  pip install qfi-opt

To install from source::

  git clone https://github.com/jmlarson1/QFI-Opt.git
  pip install -e QFI-Opt


Cite QFI-Opt
============

.. code-block:: bibtex

  @misc{qfiopt,
    author = {Colussi, V.~E. and Larson, J. and Lewis-Swan, R.~J. and 
              Narayanan, S.~H.~K. and Perlin, M.~A. and Zu\~{n}iga Castro, J.~C.},
    title  = {{QFI-Opt}},
    url    = {https://github.com/jmlarson1/QFI-Opt},
    year   = {2024}
  }

  @article{Zuniga2024,
    doi = {10.48550/arXiv.2406.01859},
    author = {Juan C. Zu\~{n}iga Castro and Jeffrey Larson and Sri Hari Krishna Narayanan 
              and Victor E. Colussi and Michael A. Perlin and Robert J. Lewis-Swan},
    title = {Variational quantum state preparation for quantum-enhanced metrology in noisy systems},
    year = {2024},
    journal = (arXiv:2406.01859},    
  }

  @article{LewisSwan2023,
    doi = {10.48550/arXiv.2311.17275},
    author = {Robert J. Lewis-Swan and Juan C. Zu\~{n}iga Castro and D. Barberena and A. M. Rey},
    title = {Exploiting nonclassical motion of a trapped ion crystal for quantum-enhanced 
             metrology of global and differential spin rotations},
    year = {2023},
    journal = (arXiv:2311.17275},    
  }




.. |qfiopt_logo| image:: https://raw.githubusercontent.com/jmlarson1/QFI-Opt/main/images/qfi-opt.png
   :align: middle
   :alt: QFI-Opt
