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
     * @brief A cuda device buffer object that represents a memory buffer on a cuda device.
     * 
     * @tparam T 
     * @note See https://www.quantstart.com/articles/Matrix-Matrix-Multiplication-on-the-GPU-with-Nvidia-CUDA/
     * @note See https://forums.developer.nvidia.com/t/guide-cudamalloc3d-and-cudaarrays/23421
     */
    template<typename T>
    class device_tensor3
    {
        size_t m_sizeDim0;
        size_t m_sizeDim1;
        size_t m_sizeDim2;
        T* m_buffer = nullptr;

    public:
        device_tensor3(size_t sizeDim0, size_t sizeDim1, size_t sizeDim2, const T* data = nullptr)
        : m_sizeDim0(sizeDim0)
        , m_sizeDim1(sizeDim1)
        , m_sizeDim2(sizeDim2)
        {
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

        device_tensor3(const Eigen::Tensor<T, 3>& tensor)
        : device_tensor3(tensor.dimension(0), tensor.dimension(1), tensor.dimension(2), tensor.data()) {}

        /**
         * @brief Copy Constructor
         * 
         * @param other 
         */
        device_tensor3(device_tensor3&& other)
            : m_sizeDim0(other.m_sizeDim0)
            , m_sizeDim1(other.m_sizeDim1)
            , m_sizeDim2(other.m_sizeDim2)
            , m_buffer(other.m_buffer)
        {
            other.m_buffer = nullptr;
            other.m_sizeDim0 = 0;
            other.m_sizeDim1 = 0;
            other.m_sizeDim2 = 0;
        }

        ~device_tensor3()
        {
            if(m_buffer != nullptr)
            {
                checkCudaErrors(cudaFree(m_buffer));
            }
        }

        __host__ __device__ T* Get()
        {
            return m_buffer;
        }

        __host__ __device__ const T* Get() const
        {
            return m_buffer;
        }

        __host__ __device__ size_t GetDimensionSize(int dim) const
        {
            if(dim == 0) return m_sizeDim0;
            if(dim == 1) return m_sizeDim1;
            if(dim == 2) return m_sizeDim2;
            return 0; //TODO: not a great interface
        }

        __host__ __device__ Eigen::DSizes<Eigen::DenseIndex, 3> GetDimensions()
        {
            auto res = Eigen::DSizes<Eigen::DenseIndex, 3>();
            res[0] = m_sizeDim0;
            res[1] = m_sizeDim1;
            res[2] = m_sizeDim2;
            return res;
        }

        __host__ __device__ size_t GetCount() const
        {
            return m_sizeDim0 * m_sizeDim1 * m_sizeDim2;
        }

        __host__ __device__ size_t GetSize() const
        {
            return GetCount();
        }

        __host__ __device__ size_t GetByteSize() const
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
            DebugCudaErrors();
        }

        /**
         * @brief Set the Data asyncronously
         * 
         * @param data 
         * @return __host__ 
         */
        __host__ void SetDataAsync(const T* data)
        {
            size_t bytes = GetByteSize();
            checkCudaErrors(cudaMemcpyAsync(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
            DebugCudaErrors();
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

        __host__ void ToHost(Eigen::Tensor<T, 3>& out) const
        {
            out.resize(GetDimensionSize(0), GetDimensionSize(1), GetDimensionSize(2));
            ToHost(out.data());
        }

        __host__ void ToHostASync(T* out) const
        {
            size_t bytes = GetByteSize();
            checkCudaErrors(cudaMemcpyAsync(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    };
}
}

#endif //CUDA_ENABLED
