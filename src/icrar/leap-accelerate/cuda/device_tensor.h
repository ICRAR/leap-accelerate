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

#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace icrar
{
namespace cuda
{
/**
     * @brief A cuda device tensor buffer object that references a tensor in cuda device memory buffer for manipulation by
     * the host.
     * 
     * @tparam T the tensor data type
     * @tparam NumDims number of tensor dimensions
     * @note See https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
     * @note See https://forums.developer.nvidia.com/t/guide-cudamalloc3d-and-cudaarrays/23421
     */
    template<typename T, uint32_t NumDims>
    class device_tensor
    {
        Eigen::DSizes<Eigen::DenseIndex, NumDims> m_sizeDim;
        T* m_buffer = nullptr;

    public:
        device_tensor(Eigen::DSizes<Eigen::DenseIndex, NumDims> shape, const T* data = nullptr)
        {
            m_sizeDim = shape;
            size_t byteSize = GetByteSize();
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

        device_tensor(size_t sizeDim0, size_t sizeDim1, size_t sizeDim2, const T* data = nullptr)
        : device_tensor({sizeDim0, sizeDim1, sizeDim2}, data)
        {
            EIGEN_STATIC_ASSERT(NumDims == 3, YOU_MADE_A_PROGRAMMING_MISTAKE);
        }
        device_tensor(size_t sizeDim0, size_t sizeDim1, size_t sizeDim2, size_t sizeDim3, const T* data = nullptr)
        : device_tensor({sizeDim0, sizeDim1, sizeDim2, sizeDim3}, data)
        {
            EIGEN_STATIC_ASSERT(NumDims == 4, YOU_MADE_A_PROGRAMMING_MISTAKE);
        }

        device_tensor(const Eigen::Tensor<T, NumDims>& tensor)
        : device_tensor(tensor.dimensions(), tensor.data()) {}

        /**
         * @brief Copy Constructor
         * 
         * @param other 
         */
        device_tensor(device_tensor&& other)
            : m_sizeDim(other.m_sizeDim)
            , m_buffer(other.m_buffer)
        {
            other.m_sizeDim = Eigen::DSizes<Eigen::DenseIndex, NumDims>();
            other.m_buffer = nullptr;
        }

        device_tensor& operator=(device_tensor&& other) noexcept
        {
            m_sizeDim = other.m_sizeDim;
            m_buffer = other.m_buffer;
            other.m_sizeDim = Eigen::DSizes<Eigen::DenseIndex, NumDims>();
            other.m_buffer = nullptr;
            return *this;
        }

        ~device_tensor()
        {
            checkCudaErrors(cudaFree(m_buffer));
        }

        /**
         * @brief Gets the raw pointer to device buffer memory
         * 
         * @return T* 
         */
        __host__ T* Get()
        {
            return m_buffer;
        }

        /**
         * @brief Gets the raw pointer to device buffer memory
         * 
         * @return T const* 
         */
        __host__ const T* Get() const
        {
            return m_buffer;
        }

        __host__ Eigen::DenseIndex GetDimensionSize(int dim) const
        {
            return m_sizeDim[dim];
        }

        __host__ Eigen::DSizes<Eigen::DenseIndex, NumDims> GetDimensions() const
        {
            return m_sizeDim;
        }

        /**
         * @brief Gets the total number of elements in the tensor
         * 
         * @return __host__ 
         */
        __host__ size_t GetCount() const
        {
            return m_sizeDim.TotalSize();
        }

        /**
         * @brief Gets the total number of elements in the tensor
         * 
         * @return __host__ 
         */
        __host__ size_t GetSize() const
        {
            return GetCount();
        }

        /**
         * @brief Gets the total number of bytes in the memory buffer 
         * 
         * @return __host__ 
         */
        __host__ size_t GetByteSize() const
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
            size_t bytes = GetByteSize();
            checkCudaErrors(cudaMemcpy(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
#ifndef NDEBUG
            DebugCudaErrors();
#endif
        }

        /**
         * @brief Set the Data asyncronously from host memory
         * 
         * @param data 
         * @return __host__ 
         */
        __host__ void SetDataAsync(const T* data)
        {
            size_t bytes = GetByteSize();
            checkCudaErrors(cudaMemcpyAsync(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }

        __host__ void SetDataAsync(const device_tensor<T, NumDims>& data)
        {
            size_t bytes = GetByteSize();
            cudaMemcpyAsync(m_buffer, data.Get(), bytes, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        }

        __host__ void ToHost(T* out) const
        {
            size_t bytes = GetByteSize();
            checkCudaErrors(cudaMemcpy(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        __host__ void ToHost(std::vector<T>& out) const
        {
            out.resize(GetSize());
            ToHost(out.data());
        }

        __host__ void ToHost(Eigen::Tensor<T, NumDims>& out) const
        {
            out.resize(GetDimensions());
            ToHost(out.data());
        }

        __host__ void ToHostAsync(T* out) const
        {
            size_t bytes = GetByteSize();
            checkCudaErrors(cudaMemcpyAsync(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    };

    template<typename T>
    using device_tensor3 = device_tensor<T, 3>;
    template<typename T>
    using device_tensor4 = device_tensor<T, 4>;
}
}

#endif //CUDA_ENABLED
