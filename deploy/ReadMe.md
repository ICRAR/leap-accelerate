# Linux Cluster Deployment

## Build

ICRAR environment setup command:

```
module load cmake/3.15.1 gcc/6.3.0 boost/1.66.0 casacore/3.1.2
module unload gfortran/default
module load isl/default
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

Build command:

```
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDA_ENABLED=TRUE -DCUDA_HOST_COMPILER=g++ -DCASACORE_ROOT_DIR=$BLDR_CASACORE_BASE_PATH -DCMAKE_BUILD_TYPE=Release
make LeapAccelerateCLI -j2
```

## Deploy Script

Alternatively, deploy to hyades03 with the following command:

```
cd deploy
./build.sh -s hyades -c /usr/local/cuda-11.0/ -D "-DCUDA_ENABLED=TRUE"
```