# leap-accelerate-lib {#leap-accelerate-lib}

A generic library containing functionality for leap calibration.

Functionality in this project is split into three independant namespaces/implementations:

* casalib
* cpu
* cuda

## casalib

Provides computation using casa math libraries for minimum overhead when
reading from casa::MeasurementSets. Casa math libraries may not contain suitable
layout for raw access by acceleration libraries such cuda, blas, cycl, etc.

## cpu

Provides computation using Eigen3 math libraries and layout that provides trivial
data layouts compatible with raw buffer copying to acceleration libraries and compatible 
with kernel compilers.

See [http://eigen.tuxfamily.org/index.php?title=Main_Page](Eigen) for more details.

## cuda
Provides Cuda computation using native cuda libraries and buffer objects.