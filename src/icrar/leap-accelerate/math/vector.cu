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

/**
 * @brief Performs vector addition of equal length vectors
 * 
 * @tparam T vector value type
 * @param x1 left vector
 * @param x2 right vector
 * @param y out vector
 * @return __global__ add 
 */
template<typename T>
__global__ void add(const T* x1, const T* x2, T* y)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}

extern "C"
{
    __global__ void addi(const int* x1, const int* x2, int* y)
    {
        add(x1, x2, y);
    }

    __global__ void addf(const float* x1, const float* x2, float* y)
    {
        add(x1, x2, y);
    }

    __global__ void addd(const double* x1, const double* x2, double* y)
    {
        add(x1, x2, y);
    }
}
