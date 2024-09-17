/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <cuda_runtime.h>

#include "cuda_info.h"
#include "helper_cuda.cuh"


#include <iostream>

int GetCudaDeviceCount()
{
    int deviceCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

void printCudaVersion()
{
#ifdef __NVCC__
    std::cout << "CUDA NVCC Compiler version: " << __CUDACC_VER_MAJOR__ << __CUDACC_VER_MINOR__ << __CUDACC_VER_BUILD__ << std::endl;
#endif

    int runtime_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}