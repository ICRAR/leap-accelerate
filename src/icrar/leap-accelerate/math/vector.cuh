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

#include <array>
#include <vector>
#include <stdexcept>

__constant__ float identity[9];

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
    y[threadId] = x1[threadId] + x2[threadId];
}

template<typename T>
__global__ void add(const T* x1, const T* x2, T* y)
{
    d_add(x1, x2, y);
}

template<typename T, int32_t S>
__host__ void h_add(const std::array<T, S>& a, const std::array<T, S>& b, std::array<T, S>& c)
{
    //8-series 128 threads
    //10-series 240 threads
    constexpr int threadsPerBlock = 1024;
    int gridSize = (int)ceil((float)S / threadsPerBlock);

    int* aBuffer;
    int* bBuffer;
    int* cBuffer;
    cudaMalloc((void**)&aBuffer, sizeof(a));
    cudaMalloc((void**)&bBuffer, sizeof(b));
    cudaMalloc((void**)&cBuffer, sizeof(c));

    cudaMemcpy(aBuffer, &a, sizeof(a), cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(bBuffer, &b, sizeof(b), cudaMemcpyKind::cudaMemcpyHostToDevice);

    add<<<gridSize, threadsPerBlock>>>(aBuffer, bBuffer, cBuffer);

    //cudaDeviceSynchronize();

    //cudaMemcpytoSymbol()
    cudaMemcpy(&c, cBuffer, sizeof(c), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(aBuffer);
    cudaFree(bBuffer);
    cudaFree(cBuffer);
}

template<typename T>
__host__ void h_add(std::vector<T> a, std::vector<T> b, std::vector<T>& c)
{
    if (a.size() != b.size() && a.size() != c.size())
    {
        throw std::runtime_error("argument sizes must be equal");
    }

    //8-series 128 threads
    //10-series 240 threads
    constexpr uint32_t threadsPerBlock = 1024;
    uint32_t S = static_cast<uint32_t>(a.size());
    uint32_t gridSize = static_cast<uint32_t>(ceil((float)S / threadsPerBlock));

    size_t byteSize = a.size() * sizeof(T);

    int* aBuffer;
    int* bBuffer;
    int* cBuffer;
    cudaMalloc((void**)&aBuffer, byteSize);
    cudaMalloc((void**)&bBuffer, byteSize);
    cudaMalloc((void**)&cBuffer, byteSize);

    cudaMemcpy(aBuffer, a.data(), byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(bBuffer, b.data(), byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

    add<<<gridSize, threadsPerBlock>>>(aBuffer, bBuffer, cBuffer);

    //cudaDeviceSynchronize();

    cudaMemcpy(c.data(), cBuffer, byteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(aBuffer);
    cudaFree(bBuffer);
    cudaFree(cBuffer);
}