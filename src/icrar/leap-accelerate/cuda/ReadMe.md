# leap-accelerate cuda

This module within leap-accelerate provides cuda acceleration for common math and algorithmic operations.

Cuda files here are compiled by nvcc for linking with leap-accelerate.

## File types

* *.cu - Cuda C++ source file targetted by nvcc.
* *.cuh - Cuda C++ header containing cuda only code. To access functions here eslewhere they must be exposed via a *.h file
* *.h - C++ compatible Cuda header. These headers may be included by both C++ and CUDA source files. 