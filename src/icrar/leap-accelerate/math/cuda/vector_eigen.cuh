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

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <casacore/casa/Arrays/Array.h>

#include <Eigen/Core>

#include <array>
#include <vector>
#include <stdexcept>

namespace icrar
{
namespace cuda
{
    /**
    * @brief Performs vector addition of equal length vectors
    *
    * @tparam T vector value type
    * @param n number of elements/vector length
    * @param x1 left vector
    * @param x2 right vector
    * @param y out vector
    */
    template<typename T, size_t N>
    __global__
    void g_add(const Eigen::Matrix<T, N, 1>& a, const Eigen::Matrix<T, N, 1>& b, Eigen::Matrix<T, N, 1>& c)
    {
        int threadId = blockDim.x * blockIdx.x + threadIdx.x;
        if(threadId < N)
        {
            c(threadId, 0) = a(threadId, 0) + b(threadId, 0);
        }
    }

    template<typename T, size_t N>
    __host__ void h_add(
        const Eigen::Matrix<T, N, 1>& a,
        const Eigen::Matrix<T, N, 1>& b,
        Eigen::Matrix<T, N, 1>& c)
    {
        size_t byteSize = sizeof(a);

        Eigen::Matrix<T, N, 1>* d_a;
        Eigen::Matrix<T, N, 1>* d_b;
        Eigen::Matrix<T, N, 1>* d_c;

        checkCudaErrors(cudaMalloc((void**)&d_a, sizeof(a)));
        checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(b)));
        checkCudaErrors(cudaMalloc((void**)&d_c, sizeof(c)));

        checkCudaErrors(cudaMemcpy((void*)d_a, (void*)&a, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)d_b, (void*)&b, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void*)d_c, (void*)&c, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
        
        // total thread count may be greater than required
        constexpr int threadsPerBlock = 1024;
        int gridSize = (int)ceil((float)N / threadsPerBlock);
        g_add<T, N><<<gridSize, threadsPerBlock>>>(*d_a, *d_b, *d_c);

        checkCudaErrors(cudaDeviceSynchronize());

        //cudaMemcpytoSymbol()
        checkCudaErrors(cudaMemcpy((void*)&c, (void*)d_c, byteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_a));
        checkCudaErrors(cudaFree(d_b));
        checkCudaErrors(cudaFree(d_c));
    }
}
}