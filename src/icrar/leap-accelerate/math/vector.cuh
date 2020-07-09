/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <casacore/casa/Arrays/Array.h>

#include <array>
#include <vector>
#include <stdexcept>

/**
* @brief Performs vector addition of equal length vectors
*
* @tparam T vector value type
* @param x1 left vector
* @param x2 right vector
* @param y out vector
*/
template<typename T>
__device__ void d_add(const T* x1, const T* x2, T* y)
{
    //1D indexing
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    //if(threadId < 0) {
        y[threadId] = x1[threadId] + x2[threadId];
    //}
}

template<typename T>
__global__ void g_add(const T* x1, const T* x2, T* y)
{
    d_add(x1, x2, y);
}

template<typename T>
__host__ void h_add(const T* a, const T* b, T* c, unsigned int n)
{
    constexpr int threadsPerBlock = 1024;
    int gridSize = (int)ceil((float)n / threadsPerBlock);

    size_t byteSize = n * sizeof(T);

    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a, byteSize);
    cudaMalloc((void**)&d_b, byteSize);
    cudaMalloc((void**)&d_c, byteSize);

    checkCudaErrors(cudaMemcpy(d_a, a, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, b, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));

    g_add<<<gridSize, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaDeviceSynchronize();

    //cudaMemcpytoSymbol()
    checkCudaErrors(cudaMemcpy(c, d_c, byteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
}

template<typename T, std::int32_t N>
__host__ void h_add(const std::array<T, N>& a, const std::array<T, N>& b, std::array<T, N>& c)
{
    h_add(a.data(), b.data(), c.data(), a.size());
}

// template<typename T, std::int32_t N>
// __host__ std::array<T, N> h_add(const std::array<T, N>& a, const std::array<T, N>& b)
// {
//     std::array<T, N> result;
//     h_add(a.data(), b.data(), result.data(), a.size());
//     return result;
// }

template<typename T>
__host__ void h_add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c)
{
    if (a.size() != b.size() && a.size() != c.size())
    {
        throw std::runtime_error("argument sizes must be equal");
    }

    h_add(a.data(), b.data(), c.data(), a.size());
}

template<typename T>
__host__ void h_add(const casacore::Array<T>& a, const casacore::Array<T>& b, casacore::Array<T>& c)
{
    if (a.shape() != b.shape() && a.shape() != c.shape())
    {
        throw std::runtime_error("argument shapes must be equal");
    }

    h_add(a.data(), b.data(), c.data(), a.shape()[0]);
}

void printCudaVersion();