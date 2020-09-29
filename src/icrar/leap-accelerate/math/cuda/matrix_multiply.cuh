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
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <casacore/casa/Arrays/Matrix.h>

namespace
{
    const unsigned int BLOCK_HEIGHT = 1024;
    const unsigned int BLOCK_WIDTH = 64;
    const unsigned int BLOCK_SIZE = 16;
}

namespace icrar
{
namespace cuda
{
    //////////////////
    // Matrix - Matrix
    
    /**
     * @brief 
     * 
     * @tparam T 
     * @param m 
     * @param n 
     * @param k 
     * @param a GPU device pointer to a m X n matrix (A)
     * @param b GPU device pointer to a n X k matrix (B)
     * @param c GPU device output purpose pointer to a m X k matrix (C)
     * @return __global__ 
     */
    template<typename T>
    __global__ void g_multiply_matrix(const int m, const int n, const int k, const T *a, const T* b, T* c)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.y * blockDim.y + threadIdx.y;

        int sum = 0;
        if (col < k && row < m)
        {
            for(int i = 0; i < n; i++) 
            {
                sum += a[row * n + i] * b[i * k + col];
            }
            c[row * k + col] = sum;
        }
    }

    template<typename T>
    __host__ void h_multiply_matrix(const int m, const int n, const int k, const T *a, const T* b, T* c)
    {
        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto dimGrid = dim3(grid_cols, grid_rows);
        auto dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
        g_multiply_matrix<<<dimGrid, dimBlock>>>(m, n, k, a, b, c);
    }

    template<typename T>
    __host__ void h_multiply(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& b,
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& c)
    {
        if(a.cols() != b.rows() || a.rows() != c.rows() || b.cols() != c.cols())
        {
            if(a.cols() != b.rows())
            {
                throw std::runtime_error("a.cols does not match b.rows");
            }
            if(a.rows() != c.rows())
            {
                throw std::runtime_error("a.rows does not match c.rows");
            }
            if(b.cols() != c.cols())
            {
                throw std::runtime_error("b.cols does not match c.cols");
            }
            throw std::runtime_error("matrix dimensions invalid");
        }

        c = a * b; //TODO use CUBlas
        //h_multiply_matrix(a.rows(), a.cols(), b.cols(), a.data(), b.data(), c.data());
    }

    template<typename T>
    __host__ void h_multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b, casacore::Matrix<T>& c)
    {
        if(a.shape()[1] != b.shape()[0] || a.shape()[0] != c.shape()[0] || b.shape()[1] != c.shape()[1])
        {
            throw std::runtime_error("matrix dimensions invalid");
        }

        m_multiply_matrix(a.shape()[0], a.shape()[1], b.shape()[1], a.data(), b.data(), c.data());
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
    __device__ void d_multiply_vector(const size_t m, const size_t n, const T* mat, const T* in, T* out)
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
    __global__ void g_multiply_vector(const size_t m, const size_t n, const T* mat, const T* vec, T* out)
    {
        //d_multiply_vector(m, n, mat, vec, out);
    }

    template<typename T>
    __host__ void h_multiply_vector(const size_t m, const size_t n, const T* mat, const T* vec, T* out)
    {
        g_multiply_vector(m, n, mat, vec, out);
    }

    template<typename T>
    __host__ void h_multiply(
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& a,
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
        Eigen::Matrix<T, Eigen::Dynamic, 1>& c)
    {
        if(a.cols() != b.rows() || b.cols() != c.cols())
        {
            throw std::runtime_error("matrix dimensions invalid");
        }

        c = a * b; //TODO: Use CuBLAS
        //h_multiply_vector(a.rows(), a.cols(), a.data(), b.data(), c.data());
    }

    template<typename T>
    __host__ void h_multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b, casacore::Vector<T>& c)
    {
        h_multiply_vector(a.shape()[0], a.shape()[1], a.data(), b.data(), c.data());
    }

    template<typename T>
    __host__ casacore::Array<T> h_multiply(const casacore::Matrix<T>& a, const casacore::Vector<T>& b)
    {
        auto c = casacore::Vector<T>();
        h_multiply(a, b, c);
        return c;
    }
}
}
