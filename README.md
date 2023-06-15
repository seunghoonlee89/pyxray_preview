
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

PyXray
===============================================

A library of ab-initio X-ray spectrum simulation using correction vector restricted active space approach

Installation
------------

* Prerequisites
    - [PySCF](https://github.com/pyscf/pyscf): 2.0 or higher.
    - [Block2](https://github.com/block-hczhai/block2-preview): p0.5.2rc7 or higher.
        (`complex mode` and `general-spin mode` are necessary. 
        `cmake` can be used to compile C++ part of Block2 code. please see Block2 manual for more details.)
        ```
        mkdir build
        cd build
        cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DLARGE_BOND=ON -DMPI=ON -DUSE_COMPLEX=ON -DUSE_SG=ON
        make -j 10
        ```

* Add PyXray top-level directory to your `PYTHONPATH`
    e.g. if pyxray_preview is installed in `/opt`, your `PYTHONPATH` should be
    ```
    export PYTHONPATH=/opt/pyxray_preview:$PYTHONPATH
    ```

Reference
------------
https://arxiv.org/abs/2305.08184
