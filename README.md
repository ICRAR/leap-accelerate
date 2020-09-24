# LEAP Accelerate

![License](https://img.shields.io/badge/license-LGPL_2.1-blue)
[![Build Status](https://travis-ci.com/ICRAR/leap-accelerate.svg?token=1YzqBsytWggkjwq3sjZP&branch=master)](https://travis-ci.com/ICRAR/leap-accelerate)

Low Frequency Excision of the atmosphere in Parallel (LEAP) Calibration using GPU acceleration.

LEAP-Accelerate includes:

* leap-accelerate: a shared library for gpu accelerated direction centering and phase calibration
* leap-accelerate-cli: a CLI interface for I/O datastream or plasma data access 
* leap-accelerate-client: a socket client interface for processing data from a LEAP-Cal server
* leap-accelerate-server: a socket server interface for dispatching data processing to LEAP-Cal clients

## Build

### Linux

`mkdir -p build/linux/{Debug,Release} && cd build/linux`

`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64`

`export PATH=$PATH:$CUDA_HOME/bin`

#### Debug

`cd Debug`

`cmake ../../../ -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" -DCMAKE_CXX_FLAGS_DEBUG=-O1 -DCMAKE_BUILD_TYPE=Debug`

#### Release

`cd Release`

`cmake ../../../ -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" -DCMAKE_BUILD_TYPE=Release`

#### Linux Cluster

`module load cmake/3.15.1 gcc/6.3.0 boost/1.66.0 casacore/3.1.2`

#### Ubuntu/Debian Dependencies

20.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.71-all-dev libeigen3-dev libgsl-dev
* https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal

or

* sudo apt-get install nvidia-cuda-toolkit-gcc

18.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.65-all-dev libeigen3-dev libgsl-dev
* https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

Note:
eigen3.3.7 is required, see .travis/before_install_bionic.sh

16.04 LTS

* https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
* sudo apt-get install gcc g++ gdb doxygen casacore-dev libboost1.58-all-dev libeigen3-dev libgsl-dev

#### Recommended Versions Compatibility

* g++ 9.3.0
* cuda 10.1
* boost 1.71.0
* casacore 3.1.2
* eigen 3.3.90

#### Minimum Versions Compatibility

* g++ 6.3.0
* cuda 9.0
* boost 1.63.0 (1.55.0 available)
* cmake 3.15.1
* casacore 3.1.2

## Test

Testing provided via googletest. To test using CTest use the following command in build/linux:

`make test` or `ctest`

for verbose output use:

`ctest --verbose`

To test using the google test runner, the test binaries can be executed directly using the following commands:

`./src/icrar/leap-accelerate/tests/LeapAccelerate.Tests`
`./src/icrar/leap-accelerate-cli/tests/LeapAccelerateCLI.Tests`

## Doxygen

Doxygen is generated with the following target:

`make doxygen`

Generated doxygen is available at the following file location:

`src/out/html/index.html`

## Run CLI

`./src/icrar/leap-accelerate-cli/LeapAccelerateCLI`

or

`./bin/LeapAccelerateCLI`

## Profiling

* nvprof
* gprof
* google-perftools

