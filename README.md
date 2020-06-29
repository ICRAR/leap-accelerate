# LEAP Accelerate

Acceleration module for LEAP using GPU acceleration. LEAP-Accelerate includes:

* leap-accelerate: a shared library for gpu accelerated direction centering and phase calibration
* leap-accelerate-cli: a CLI interface for I/O datastream or plasma data access 
* leap-accelerate-client: a socket client interface for recieving and sending to LEAP-Cal

## Building

### Linux

`mkdir build && cd build`

`mdir linux && cd linux`

`cmake ../../`

## Testing

Testing provided via googletest.

in build directory run:

`make test`

## Run Server

`./build/linux/src/icrar/leap-accelerate/`

TODO:
local installation to a single bin directory

`./build/linux/bin/LeapAccelerate`

## Debian Dependencies
* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10