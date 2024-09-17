# LEAP Accelerate Python

![License](https://img.shields.io/badge/license-GPLv2+-blue)

**leap-accelerate** is a calibration tool implementing Low-frequency Excision of the Atmosphere in Parallel ([LEAP](https://arxiv.org/abs/1807.04685)) for low-frequency radio antenna arrays. Leap utilizes GPGPU acceleration for parallel computation across baselines, channels and polarizations and is freely available on [GitLab](https://gitlab.com/ska-telescope/icrar-leap-accelerate) under GPLv2+ [License](LICENSE).

leap-accelerate includes:

* [leap-accelerate](../README.md): a shared library for gpu accelerated direction centering and phase calibration.
* [leap-accelerate-cli](src/icrar/leap-accelerate-cli/ReadMe.md): a native CLI to leap-accelerate with file output.
* **leap-accelerate-python**: python3 bindings library and CLI to leap-accelerate using pybind11.
<!---* leap-accelerate-client: a socket client interface for processing data from a LEAP-Cal server--->
<!---* leap-accelerate-server: a socket server interface for dispatching data processing to LEAP-Cal clients--->

See the [online documentation](https://developer.skatelescope.org/projects/icrar-leap-accelerate/en/latest/) for more information.

## Usage (API)

```python
import leap
import tempfile
import json
import numpy as np

calibrator = leap.LeapCalibrator("cpu")

output = list()
calibrator.calibrate(
    ms=leap.MeasurementSet("../testdata/mwa/1197638568-split.ms"),
    directions=np.array([[0.1,0.2],[0.3, 0.4],[0.5,0.6]]),
    callback=output.append)

print(output)
```

## Usage (CLI)

```bash
leap_cli batch plot ../testdata/mwa/1197638568-split4-dropped.ms/ -d "[[7.09767229e-01, -1.98773609e-04]]" -m 120m
```

## Installation

### Poetry Environment

It is recommended to setup a poetry environment for building (but not required, see end user environment instructions):

```bash
poetry env use <python-version>

poetry shell

# use same casacore binary as native leap-accelerate
pip install python-casacore --no-binary python-casacore
```

#### Build and Install

developer install to poetry environment:

```bash
# stable system build and install
CUDA_ENABLED=1 poetry install

# or build and install with specific compiler
CMAKE_CXX_COMPILER=/usr/bin/g++-12 CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc CUDA_ENABLED=1 poetry install
```

or test the production installation using poetry:

```bash
# stable system build
CUDA_ENABLED=1 poetry build -vv

# or build with specific compiler
CMAKE_CXX_COMPILER=/usr/bin/g++-12 CMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc CUDA_ENABLED=1 poetry build -vv

# install pre-built binary
pip install dist/leap-0.13.0-cp311-cp311-manylinux_2_38_x86_64.whl
```

#### Test

```bash
poetry install --only test
pytest
```

### End User Environment

```bash

# use same libcasa_ms.so library as native leap-accelerate
# otherwise will experience errors such as
# free(): invalid pointer
pip install python-casacore --no-binary python-casacore

# build without poetry
python -m build -vv

# install pre-built binary
pip install dist/leap-0.13.0-cp311-cp311-manylinux_2_38_x86_64.whl
```
