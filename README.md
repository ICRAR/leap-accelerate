# LEAP Accelerate

![License](https://img.shields.io/badge/license-LGPL_2.1-blue)
[![Build Status](https://travis-ci.com/ICRAR/leap-accelerate.svg?token=1YzqBsytWggkjwq3sjZP&branch=master)](https://travis-ci.com/ICRAR/leap-accelerate)

Low-frequency Excision of the Atmosphere in Parallel (LEAP) Calibration using GPU acceleration.

LEAP-Accelerate includes:

* leap-accelerate: a shared library for gpu accelerated direction centering and phase calibration
* leap-accelerate-cli: a CLI interface for I/O datastream or plasma data access 
* leap-accelerate-client: a socket client interface for processing data from a LEAP-Cal server
* leap-accelerate-server: a socket server interface for dispatching data processing to LEAP-Cal clients

## Docker image build
Due to the size of the CUDA tool chain the build of a LEAP_accelerate docker image has been split into two parts, but there is also a Dockerfile.bootstrap which does most of the build in one go. Trimming the final image down to a reasonable size is another required step.

### Bootstrap build
The Dockerfile.bootstrap builds the image from scratch, but that takes pretty long. NOTE: Depending on the network connection this build can take a long time. It is downloading the CUDA tool chain which is about 2.7 GB. After the download the unpacking and installation takes significant time in addition.

`docker build --tag icrar/leap_cli:big -f Dockerfile.bootstrap`

NOTE: Replace the `<version>` with the version of leap-accelarate this image had been build from.

### Separate build
The Dockerfile.base builds an image with the CUDA toolchain, which is the base for the actual leap_accelarate image and needs to be build far less often than the leap_accelarate.

`docker build --tag 20.04cuda:installed -f Dockerfile.base`

To get the leap_accelarate image run

`docker build . --tag icrar/leap_cli:big`

### Stripping the image

That image is still very large (>5GB). In order to clean this up it is possible to run the tool from https://github.com/mvanholsteijn/strip-docker-image.git

`strip-docker-image -i icrar/leap_cli:big -t icrar/leap_cli:<version> -f /usr/local/bin/LeapAccelerateCLI -f /usr/bin/sh`

NOTE: Replace the `<version>` with the version of leap-accelarate this image had been build from.

### Test the image
Run install.sh in the testdata directory and then in the main directory of leap_accelarate:

`docker run -v "$(pwd)"/testdata:/testdata icrar/leap_cli:0.4 LeapAccelerateCLI -f /testdata/1197638568-split.ms -s 126 -i eigen -d "[[-0.4606549305661674,-0.29719233792392513]]"`

The output should be a JSON data structure.

## Build

### Linux

`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64`

`export PATH=$PATH:$CUDA_HOME/bin`

`mkdir -p build/linux/{Debug,Release} && cd build/linux`

#### Debug

`cd Debug`

`cmake ../../ -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCMAKE_BUILD_TYPE=Debug`

(with tracing):

`cmake ../../ -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DTRACE=ON -DCMAKE_BUILD_TYPE=Debug`

#### Release

`cd Release`

`cmake ../../ -DCMAKE_BUILD_TYPE=Release`

#### Linux Cluster

`module load cmake/3.15.1 gcc/6.3.0 boost/1.66.0 casacore/3.1.2`

`module unload gfortran/default`

`module load isl/default`

`export CUDA_HOME=/usr/local/cuda`

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64`

`export PATH=$PATH:$CUDA_HOME/bin`

`mkdir -p build && cd build`

`cmake ../../ -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDA_HOST_COMPILER=g++ -DCASACORE_ROOT_DIR=$BLDR_CASACORE_BASE_PATH -DCMAKE_BUILD_TYPE=Release`

#### Ubuntu/Debian Dependencies

20.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.71-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal

or

* sudo apt-get install nvidia-cuda-toolkit-gcc

18.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.65-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

16.04 LTS

* https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
* sudo apt-get install gcc g++ gdb doxygen casacore-dev libboost1.58-all-dev libgsl-dev

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

