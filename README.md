# LEAP Accelerate

![License](https://img.shields.io/badge/license-GPLv2+-blue)

**leap-accelerate** is a calibration tool implementing Low-frequency Excision of the Atmosphere in Parallel ([LEAP](https://arxiv.org/abs/1807.04685)) for low-frequency radio antenna arrays. Leap utilizes GPGPU acceleration for parallel computation across baselines, channels and polarizations and is freely available on [GitLab](https://gitlab.com/ska-telescope/icrar-leap-accelerate) under GPLv2+ [License](LICENSE).

leap-accelerate includes:

* **leap-accelerate**: a shared library for gpu accelerated direction centering and phase calibration.
* [leap-accelerate-cli](src/icrar/leap-accelerate-cli/README.md): a CLI interface for I/O datastream or plasma data access.
* [leap-accelerate-python](python/README.md): python3 bindings to leap-accelerate using pybind11.
<!---* leap-accelerate-client: a socket client interface for processing data from a LEAP-Cal server--->
<!---* leap-accelerate-server: a socket server interface for dispatching data processing to LEAP-Cal clients--->

See the [online documentation](https://developer.skatelescope.org/projects/icrar-leap-accelerate/en/latest/) for more information.

## Installation

### leap-accelerate and leap-accelerate-cli

#### Build

```bash
cmake -B build -G "Ninja Multi-Config"
cmake --build build --config Release -j8
```

#### Test

```bash
ctest --test-dir build -C Release -T test --verbose
```

#### Install

```bash
cmake --install build --config Release
```

## OSI Image

The latest leap release is published as a debian a docker image available at the following location:

`artefact.skao.int/icrar-leap-accelerate:latest`

NOTE: It may be necessary to use the image tag rather than just 'latest'.

This image can be run locally using the following command:

`docker run -it --rm artefact.skao.int/icrar-leap-accelerate:latest LeapAccelerateCLI --help`

See the [docker](docs/src/md/Docker.md) documentation for instructions about how to create a docker image.

See the [build](docs/src/md/Build.md) documentation for instructions on platform specific compilation.

## Usage

See [leap-accelerate-cli](docs/src/md/LeapAccelerateCLI.md) for instructions on command line arguments and configuration files.

Examples:

`LeapAccelerateCLI --help`

`LeapAccelerateCLI --config "./askap.json"`

## Contributions

Refer to the following style guides for making repository contributions

* [CMake Style Guide](docs/src/md/specs/CMakeStyleGuide.md)
* [C++ Style Guide](docs/src/md/specs/CPlusPlusStyleGuide.md)
