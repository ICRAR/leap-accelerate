# Compiling from Source

leap-accelerate compilation is compatible with g++ and clang++ on debian or ubuntu. Support for macOS is currently experimental.

## Binaries

### Dependencies

#### Recommended Versions Compatibility

* g++ 9.3.0
* cuda 10.1
* boost 1.71.0
* casacore 3.1.2

#### Minimum Versions Compatibility

* g++ 6.3.0
* cuda 9.0
* boost 1.63.0
* cmake 3.15.1
* casacore 3.1.2

#### Ubuntu/Debian Dependencies

##### 20.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.71-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=deblocal

##### 18.04 LTS

* sudo apt-get install gcc g++ gdb doxygen cmake casacore-dev clang-tidy-10 libboost1.65-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

##### 16.04 LTS

* https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
* sudo apt-get install gcc-6 g++-6 gdb doxygen casacore-dev libboost1.58-all-dev libgsl-dev
* https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal

#### CMake Options

Use `cmake .. -D<OPTION>=<VALUE> ...` or `ccmake ..` to set cmake options.

Setting an environment variable of the same name will also override these cmake options:

`CUDA_ENABLED` - Enables building with cuda support

`CMAKE_CUDA_ARCHITECTURES` - Selects the target cuda streaming multiprocessor and compute levels (default is all)

`WERROR` - Enables warnings as Errors

`WCONVERSION` - Enables warnings on implicit numeric conversions

`TRACE` - Traces data to the local directory

`CMAKE_RUN_CLANG_TIDY` - Enables running clang-tidy with the compiler

`USE_PCH` - Use pre-compile headers internally, if possible (defaults to `ON`)

#### Compile Commands

From the repository root folder run:

`git submodule update --init --recursive`

NOTE: pulling exernal submodules is now automated by CMake. When downloading the source files via tools other than git the folder `external/` will need to be copied manually.

#### GCC

###### Build Debug

`cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCUDA_ENABLED=TRUE`

With tracing to file:

`cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCUDA_ENABLED=TRUE -DTRACE=ON`

With gcovr analysis:

`cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCMAKE_CXX_FLAGS="-coverage" -DCMAKE_EXE_LINKER_FLAGS="-coverage"`

###### Build Release

`cmake -B build/Release -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=TRUE`

###### CUDA Hints

If cmake fails to detect CUDA try adding the following hint variables:

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

## Testing

Testing provided via googletest. To test using the google test runner, test binaries can be executed directly using the following commands from the output folder:

`./bin/tests/LeapAccelerate.Tests`
`./bin/tests/LeapAccelerateCLI.Tests`

To test using CTest use the following command in build/linux:

`make test` or `ctest`

for verbose output use:

`ctest --verbose` or `ctest --output-on-failure`

## Documentation

Generated documentation is available locally at the following file location:

`docs/sphinx/index.html`

Once deployed to a branch the docs will be available here:

https://developer.skao.int/projects/icrar-leap-accelerate/en/latest/

## Test Coverage (Debug Only)

`cmake ../.. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-g -O1" -DCMAKE_CXX_FLAGS="-coverage" -DCMAKE_EXE_LINKER_FLAGS="-coverage"`

`make coverage`

### Building from Source

Doxygen documentation is generated for all C++ and cuda files with the following target:

`make doxygen`

Sphinx/Breath/Exhale docs is a dependent target generated with the following command:

`make sphinx`

### gitlab repo

The CI/CD on gitlab used a pre-built base build image along with a cpp_build_base image to speed this process.
