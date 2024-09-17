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

#ifdef CUDA_ENABLED

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
     * @brief A cuda device buffer object that represents a global memory buffer on a cuda device. Matrix size is fixed
     * at construction and can only be resized using move semantics.
     * 
     * @tparam T numeric type
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
         * @brief Default constructor
         * 
         */
        device_matrix()
        : m_rows(0)
        , m_cols(0)
        , m_buffer(nullptr)
        { }

        /**
         * @brief Move Constructor
         * 
         * @param other 
         */
        device_matrix(device_matrix&& other) noexcept 
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_buffer(other.m_buffer)
        {
            other.m_rows = 0;
            other.m_cols = 0;
            other.m_buffer = nullptr;
        }

        /**
         * @brief Move Assignment Operator
         * 
         * @param other 
         * @return device_matrix& 
         */
        device_matrix& operator=(device_matrix&& other) noexcept
        {
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_buffer = other.m_buffer;
            other.m_rows = 0;
            other.m_cols = 0;
            other.m_buffer = nullptr;
            return *this;
        }

        /**
         * @brief Construct a new device matrix object of fixed size
         * and initialized asyncronously if data is provided
         * 
         * @param rows number of rows
         * @param cols number of columns
         * @param data constigous column major data of size rows*cols*sizeof(T) to copy to device
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

        explicit device_matrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& data)
        : device_matrix(data.rows(), data.cols(), data.data()) {}

        template<int Rows, int Cols>
        explicit device_matrix(const Eigen::Matrix<T, Rows, Cols>& data)
        : device_matrix(Rows, Cols, data.data()) {}

        ~device_matrix()
        {
            checkCudaErrors(cudaFree(m_buffer));
        }

        __host__ T* Get()
        {
            return m_buffer;
        }

        __host__ const T* Get() const
        {
            return m_buffer;
        }

        __host__ size_t GetRows() const
        {
            return m_rows;
        }

        __host__ size_t GetCols() const
        {
            return m_cols;
        }

        __host__ size_t GetCount() const
        {
            return m_rows * m_cols;
        }

        __host__ size_t GetSize() const
        {
            return GetCount() * sizeof(T);
        }

        __host__ void SetZeroAsync()
        {
            size_t byteSize = GetSize();
            checkCudaErrors(cudaMemsetAsync(m_buffer, 0, byteSize));
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
#ifndef NDEBUG
            DebugCudaErrors();
#endif
        }

        /**
         * @brief Copies data from device to host memory
         * 
         * @param data 
         * @return __host__ 
         */
        __host__ void SetDataAsync(const T* data)
        {
            size_t bytes = GetSize();
            checkCudaErrors(cudaMemcpyAsync(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }

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

        __host__ Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ToHost() const
        {
            auto result = Eigen::MatrixXd(GetRows(), GetCols());
            ToHost(result.data());
            return result;
        }

        __host__ void ToHostAsync(T* out) const
        {
            size_t bytes = GetSize();
            checkCudaErrors(cudaMemcpyAsync(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        __host__ void ToHostAsync(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& out) const
        {
            out.resize(GetRows(), GetCols());
            ToHostAsync(out.data());
        }

        __host__ void ToHostVectorAsync(Eigen::Matrix<T, Eigen::Dynamic, 1>& out) const
        {
            if(GetCols() != 1)
            {
                throw std::runtime_error("columns not 1");
            }
            out.resize(GetRows());
            ToHostAsync(out.data());
        }

        __host__ Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ToHostAsync() const
        {
            auto result = Eigen::MatrixXd(GetRows(), GetCols());
            ToHostAsync(result.data());
            return result;
        }
    };
}
}

#endif //CUDA_ENABLED
