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
