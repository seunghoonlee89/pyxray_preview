PyXray
===============================================

A library of X-ray spectrum simulation

Installation
------------

* Prerequisites
    - [PySCF](https://github.com/pyscf/pyscf): 2.0 or higher.
    - [Block2](https://github.com/block-hczhai/block2-preview): complex mode and general spin mode. cmake (version >= 3.0) can be used to compile C++ part of Block2 code, as follows:
        ```
        mkdir build
        cd build
        cmake .. -DUSE_MKL=ON -DBUILD_LIB=ON -DLARGE_BOND=ON -DMPI=ON -DUSE_COMPLEX=ON -DUSE_SG=ON
        make -j 10
        ```
        (see Block2 manual for more details)

Reference
------------
https://arxiv.org/abs/2305.08184
