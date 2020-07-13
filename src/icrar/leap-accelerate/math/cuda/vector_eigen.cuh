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
#include <eigen3/Eigen/Core>

#include <array>
#include <vector>
#include <stdexcept>

/**
* @brief Performs vector addition of equal length vectors
*
* @tparam T vector value type
* @param n number of elements/vector length
* @param x1 left vector
* @param x2 right vector
* @param y out vector
*/
template<typename T>
__device__ void d_add(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>* a,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>* b,
    Eigen::Matrix<T, Eigen::Dynamic, 1> c)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadId < n)
    {
        a(threadId, 1) = x1(threadId, 1) + x2(threadId, 1);
    }
}

template<typename T>
__global__ void d_add(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>* a,
    const Eigen::Matrix<T, Eigen::Dynamic, 1>* b,
    Eigen::Matrix<T, Eigen::Dynamic, 1> c)
{
    d_add(n, x1, x2, y);
}