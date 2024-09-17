# CUDA/C++ Style Guide

## File Structure

* All C++ headers (.h, .hpp) must be includable in sources built without cuda support
* Use C++ source files (.cc, .cpp) where possible for improved compilation speed
* Use Cuda source files (.cu) only for code blocks containing device code (at least 1 \__device__ or \__global__ definition)
* Use empty \__host__ and \__device__ definition guards in function headers to make them portable for builds without cuda support
* Do not declare \__global__ functions in C++ header or source file (.h, .hpp, .cc, .cpp)
* Declare kernel calling functions in C++ headers (.h, .hpp) and encapsulate pointers to device and host memory locations

## Naming Conventions

* Use `g_` prefix for \__global__ functions to signify a cuda kernel entry point
