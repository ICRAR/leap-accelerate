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
#include <icrar/leap-accelerate/cuda/cuda_utils.cuh>

#include <boost/noncopyable.hpp>

#include <vector>

namespace icrar
{
namespace cuda
{
    /**
     * @brief An object that represents a matrix memory buffer on a cuda device.
     * 
     * @tparam T 
     * @note See https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
     * @note See https://forums.developer.nvidia.com/t/guide-cudamalloc3d-and-cudaarrays/23421
     */
    template<typename T>
    class device_matrix : boost::noncopyable
    {
        size_t m_rows;
        size_t m_cols;
        T* m_buffer;

    public:

        /**
         * @brief Construct a new device buffer object of shape <code>[rows, cols]</code> (column major)
         * 
         * @param rows number of rows to allocate
         * @param cols number of cols to allocate
         * @param data raw pointer to buffer to initialize with, uninitialized buffer if null
         */
        device_matrix(size_t rows, size_t cols, const T* data = nullptr)
        : m_rows(rows)
        , m_cols(cols)
        , m_buffer(nullptr)
        {
            size_t byteSize = rows * cols * sizeof(T);
            checkCudaErrors(cudaMalloc((void**)&m_buffer, byteSize));
            if (data != nullptr)
            {
                checkCudaErrors(cudaMemcpyAsync(m_buffer, data, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
            }
            else
            {
                checkCudaErrors(cudaMemsetAsync(m_buffer, 0, byteSize));
            }
        }

        /**
         * @brief Construct a new device matrix object identical to the provided matrix
         * 
         * @param data 
         */
        device_matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data)
        : device_matrix(data.rows(), data.cols(), data.data()) {}

        /**
         * @brief Construct a new device matrix object identical to the provided matrix
         * 
         * @param data 
         */
        template<int Rows, int Cols>
        device_matrix(Eigen::Matrix<T, Rows, Cols> data)
        : device_matrix(Rows, Cols, data.data()) {}

        /**
         * @brief Copy Constructor
         * 
         * @param other 
         */
        device_matrix(device_matrix&& other)
            : m_rows(other.m_rows)
            , m_cols(other.m_cols)
            , m_buffer(other.m_buffer)
        {
            other.m_buffer = nullptr;
            other.m_rows = 0;
            other.m_cols = 0;
        }

        ~device_matrix()
        {
            if(m_buffer != nullptr)
            {
                checkCudaErrors(cudaFree(m_buffer));
            }
        }

        /**
         * @brief Gets the pointer the gpu memory address 
         * 
         * @return pointer to the buffer gpu memory address 
         */
        __host__ __device__ T* Get()
        {
            return m_buffer;
        }

        /**
         * @see device_matrix::Get
         */
        __host__ __device__ const T* Get() const
        {
            return m_buffer;
        }

        /**
         * @brief Get the number of rows in the buffer
         * 
         * @return number of rows in the buffer
         */
        __host__ __device__ size_t GetRows() const
        {
            return m_rows;
        }

        /**
         * @brief Get the number of cols in the buffer
         * 
         * @return number of cols in the buffer
         */
        __host__ __device__ size_t GetCols() const
        {
            return m_cols;
        }

        /**
         * @brief Gets the total number elements in the matrix buffer (e.g. rows * cols)
         * 
         * @return total number of elements
         */
        __host__ __device__ size_t GetCount() const
        {
            return m_rows * m_cols;
        }

        /**
         * @brief Gets the total number of bytes used by the matrix buffer
         * 
         * @return total number of bytes 
         */
        __host__ __device__ size_t GetSize() const
        {
            return GetCount() * sizeof(T);
        }

        /**
         * @brief Performs a synchronous copy of data into the device buffer
         * 
         * @param data 
         * @return __host__ 
         */
        __host__ void SetDataSync(const T* data)
        {
            size_t bytes = GetSize();
            checkCudaErrors(cudaMemcpy(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
            DebugCudaErrors();
        }

        /**
         * @brief Set the Data Async object
         * 
         * @param data 
         * @return __host__ 
         */
        __host__ void SetDataAsync(const T* data)
        {
            size_t bytes = GetSize();
            checkCudaErrors(cudaMemcpyAsync(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
            DebugCudaErrors();
        }

        /**
         * @brief Copies data from the device buffer to host (cpu) memory
         * 
         * @param out preallocated memory of at least @ref device_matrix::GetSize
         * @return __host__ 
         */
        __host__ void ToHost(T* out) const
        {
            size_t bytes = GetSize();
            checkCudaErrors(cudaMemcpy(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        __host__ void ToHost(std::vector<T>& out) const
        {
            out.resize(GetRows(), GetCols());
            ToHost(out.data());
        }

        __host__ void ToHost(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& out) const
        {
            out.resize(GetRows(), GetCols());
            ToHost(out.data());
        }

        template<int Rows, int Cols>
        __host__ void ToHost(Eigen::Matrix<T, Rows, Cols>& out) const
        {
            out.resize(GetRows(), GetCols());
            ToHost(out.data());
        }

        __host__ void ToHostASync(T* out) const
        {
            size_t bytes = GetSize();
            checkCudaErrors(cudaMemcpyAsync(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    };
}
}