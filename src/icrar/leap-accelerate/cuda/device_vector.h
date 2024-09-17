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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/cuda/cuda_utils.cuh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <boost/noncopyable.hpp>

#include <vector>
#include <iostream>

namespace icrar
{
namespace cuda
{
    /**
     * @brief A cuda device buffer object that own a vector memory buffer on a cuda device. Vector size is fixed
     * at construction.
     * 
     * @tparam T numeric type
     * @note See https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
     * @note See https://forums.developer.nvidia.com/t/guide-cudamalloc3d-and-cudaarrays/23421
     */
    template<typename T>
    class device_vector : boost::noncopyable
    {
        size_t m_count;
        T* m_buffer = nullptr; // Pointer to cuda owned memory

    public:
        //device_vector(const device_vector&) = delete;
        //device_vector& operator=(const device_vector&) = delete;


        /**
         * @brief Default constructor
         */
        device_vector()
        : m_count(0)
        , m_buffer(nullptr)
        { }

        __host__ device_vector(device_vector&& other) noexcept
            : m_count(other.m_count)
            , m_buffer(other.m_buffer)
        {
            other.m_count = 0;
            other.m_buffer = nullptr;
        }

        __host__ device_vector& operator=(device_vector&& other) noexcept
        {
            m_count = other.m_count;
            m_buffer = other.m_buffer;
            other.m_count = 0;
            other.m_buffer = nullptr;
            return *this;
        }

        /**
         * @brief Construct a new device buffer object
         * 
         * @param size 
         * @param data 
         */
        __host__ explicit device_vector(size_t count, const T* data = nullptr)
        : m_count(count)
        , m_buffer(nullptr)
        {
            size_t byteSize = count * sizeof(T);
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

        __host__ explicit device_vector(const std::vector<T>& data) : device_vector(data.size(), data.data()) {}

        __host__ explicit device_vector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& data) : device_vector(data.size(), data.data()) {}

        __host__ explicit device_vector(const Eigen::Matrix<T, 1, Eigen::Dynamic>& data) : device_vector(data.size(), data.data()) {}

        __host__ ~device_vector()
        {
            checkCudaErrors(cudaFree(m_buffer));
        }

        __host__ __device__ T* Get()
        {
            return m_buffer;
        }

        __host__ __device__ const T* Get() const
        {
            return m_buffer;
        }

        /**
         * @brief Gets the number of elements in the buffer
         */
        __host__ __device__ size_t GetCount() const
        {
            return m_count;
        }

        /**
         * @brief Gets the number of rows in the column vector
         */
        __host__ __device__ size_t GetRows() const
        {
            return m_count;
        }

        constexpr __host__ __device__ size_t GetCols() const
        {
            return 1;
        }

        /**
         * @brief Gets the buffer size in bytes
         */
        __host__ __device__ size_t GetSize() const
        {
            return m_count * sizeof(T);
        }

        __host__ void SetZeroAsync()
        {
            size_t byteSize = m_count * sizeof(T);
            checkCudaErrors(cudaMemsetAsync(m_buffer, 0, byteSize));
        }

        /**
         * @brief Performs a synchronous copy of data into the device buffer
         * @pre data points to a buffer of byte size >= GetSize()
         * @param data data buffer for host to device copying
         */
        __host__ void SetDataSync(const T* data)
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpy(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
#ifndef NDEBUG
            DebugCudaErrors();
#endif
        }

        /**
         * @brief Sets buffer data from pinned host memory.
         * @pre Heap memory must be pinned using cudaHostRegister(..., cudaHostRegisterPortable)
         * @pre data points to a buffer of byte size >= GetSize()
         * @param data data buffer for host to device copying
         */
        __host__ void SetDataAsync(const T* data)
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpyAsync(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }

        __host__ void ToHost(T* out) const
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpy(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        __host__ void ToHost(std::vector<T>& out) const
        {
            out.resize(GetCount());
            ToHost(out.data());
        }

        __host__ void ToHost(Eigen::Matrix<T, Eigen::Dynamic, 1>& out) const
        {
            out.resize(GetCount());
            ToHost(out.data());
        }

        /**
         * @brief Sets buffer data from pinned host memory.
         * @pre Heap memory must be pinned using cudaHostRegister(..., cudaHostRegisterPortable)
         * @pre data points to a buffer of byte size >= GetSize()
         * @param data data buffer for device to host copying
         */
        __host__ void ToHostAsync(T* out) const
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpyAsync(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        __host__ void ToHostAsync(std::vector<T>& out) const
        {
            out.resize(GetCount());
            ToHostAsync(out.data());
        }

        __host__ void ToHostAsync(Eigen::Matrix<T, Eigen::Dynamic, 1>& out) const
        {
            out.resize(GetCount());
            ToHostAsync(out.data());
        }

        __host__ Eigen::Matrix<T, Eigen::Dynamic, 1> ToHostAsync() const
        {
            auto result = Eigen::Matrix<T, Eigen::Dynamic, 1>(GetRows(), 1);
            ToHostAsync(result.data());
            return result;
        }
    };
}
}

#endif //CUDA_ENABLED
