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

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/math/eigen_helper.h>

#include <casacore/casa/Arrays/Matrix.h>

namespace
{
    const unsigned int BLOCK_HEIGHT = 1024;
    const unsigned int BLOCK_WIDTH = 64;
}

namespace icrar
{
namespace cuda
{
    //////////////////
    // Matrix - Matrix
    template<typename T>
    __global__ void d_multiply_matrix(const int m, const int n, const T *a, const T* b, T* c)
    {

    }

    template<typename T>
    __host__ void h_multiply_matrix(const int m, const int n, const T *a, const T* b, T* c)
    {
        d_multiply_matrix<<<1,1>>>(m, n, a, b, c);
    }

    template<typename T>
    __host__ void h_multiply(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& c)
    {
        c = a * b; //TODO: use kernel
    }

    template<typename T>
    __host__ void h_multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b, casacore::Matrix<T>& c)
    {
        //TODO
    }

    template<typename T>
    __host__ casacore::Matrix<T> h_multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b)
    {
        casacore::Matrix<T> c;
        h_multiply(a, b, c);
        return c;
    }



    //////////////////
    // Matrix - Vector

    /**
     * @brief 
     * 
     * @tparam T 
     * @param m matrix rows
     * @param n matrix columns
     * @param mat left side matrix
     * @param in right side column vector
     * @param out resulting column vector
     * @return __global__ 
     */
    template<typename T>
    __device__ void d_multiply_vector(const int m, const int n, const T* mat, const T* in, T* out)
    {
        // https://github.com/uysalere/cuda-matrix-vector-multiplication/blob/master/mult_kernels.cu
        __shared__ int blockElt;
        __shared__ int blockxInd;
        __shared__ int blockyInd;

        if (threadIdx.x == 0)
        {
            if ((blockIdx.y + 1) * BLOCK_WIDTH <= m)
            {
                blockElt = BLOCK_WIDTH;
            }
            else
            {
                blockElt = m % BLOCK_WIDTH;
            }
            blockxInd = blockIdx.x * BLOCK_HEIGHT;
            blockyInd = blockIdx.y * BLOCK_WIDTH;
        }

        __syncthreads();
  
        // copy section of b into shared mem
        // use the first BLOCK_WIDTH of thread
        __shared__ T b[BLOCK_WIDTH];

        if (threadIdx.x < blockElt)
            b[threadIdx.x] = in[blockyInd + threadIdx.x];
        
        __syncthreads();

        // summing variable
        T cSum = (T)0;
        int threadxInd = blockxInd + threadIdx.x;

        // make sure we are inside the array horizontally
        if (threadxInd < n) {
        
            // go through the threads vertically and sum them into a variable
            for (int i=0; i<blockElt; i++)
            {
                cSum += b[i] * mat[(threadxInd) * (m) + (blockyInd + i)];
            }
            // atomic add these variables to the corresponding c index
            atomicAdd(out + threadxInd , cSum);
        }
    }

    template<typename T>
    __global__ void g_multiply_vector(const int m, const int n, const T* mat, const T* vec, T* out)
    {
        g_multiply_vector(m, n, mat, vec, out);
    }

    template<typename T>
    __host__ void h_multiply(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
        Eigen::Matrix<T, Eigen::Dynamic, 1>& c)
    {
        c = a * b; //TODO: use kernel
    }

    template<typename T>
    __host__ void h_multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b, casacore::Array<T>& c)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ea = ConvertMatrix(a);
        Eigen::Matrix<T, Eigen::Dynamic, 1> eb = ConvertVector(b);
        Eigen::Matrix<T, Eigen::Dynamic, 1> ec = ConvertVector(c);
        h_multiply(ea, eb, ec);
        c = ConvertVector(ec);
    }

    template<typename T>
    __host__ casacore::Array<T> h_multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b)
    {
        auto c = casacore::Array<T>();
        h_multiply(a, b, c);
        return c;
    }
}
}
