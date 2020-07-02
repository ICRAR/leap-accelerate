# LEAP Accelerate

Acceleration module for LEAP using GPU acceleration. LEAP-Accelerate includes:

* leap-accelerate: a shared library for gpu accelerated direction centering and phase calibration
* leap-accelerate-cli: a CLI interface for I/O datastream or plasma data access 
* leap-accelerate-client: a socket client interface for recieving and sending to LEAP-Cal

## Build

### Linux

`mkdir build && cd build`

`mdir linux && cd linux`

`cmake ../../`

## Test

Testing provided via googletest. To test using CTest use the following command in build/linux:

`make test`

To test using the google test runner, use the following command:

`./src/icrar/leap-accelerate/tests/LeapAccelerate.Tests`

## Doxygen

Doxygen is generated with the following target:

`make doxygen`

Generated doxygen is available at the following file location:

`src/out/html/index.html`

## Run CLI

`./src/icrar/leap-accelerate-cli/LeapAccelerateCLI`

or

`./bin/LeapAccelerateCLI`

## Debian Dependencies

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.37-dev-all nvidia-cuda-toolkit-gcc

## Minimum Version Compatibility

* gcc 6.3.0
* cuda 10.1
* boost 1.66.0 (1.63.0 or 1.55.0 available)
* cmake 3.15.1
* casacore 3.1.2

module help

* module load
* module list

## Profiling
gprof
Perf tools
