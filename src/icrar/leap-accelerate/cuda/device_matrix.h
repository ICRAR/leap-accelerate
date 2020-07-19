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

#include <vector>

#ifdef DEBUG_CUDA_ERRORS
static void DebugCudaErrors()
{
    //Synchronize to make sure that any currently executing or queued for execution operations
    //that may cause errors are complete before we query the last error.
    //The synchronize may return an error code if pending operations encounter errors but will
    //not return an error code for operations that have already completed.
    CHECK_CUDA_ERROR_CODE(cudaDeviceSynchronize());
    //Query the most recent error and check the result
    CHECK_CUDA_ERROR_CODE(cudaGetLastError());
}
#else
static void DebugCudaErrors() {}
#endif


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
    class device_matrix
    {
        size_t m_count;
        T* m_buffer = nullptr;

    public:

        /**
         * @brief Construct a new device buffer object
         * 
         * @param size 
         * @param data 
         */
        device_matrix(size_t count, const T* data = nullptr)
        : m_count(count)
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

        device_matrix(std::vector<T> data) : device_matrix(data.size(), data.data()) {}

        /**
         * @brief Copy Constructor
         * 
         * @param other 
         */
        device_matrix(device_matrix&& other)
            : m_buffer(other.m_buffer)
            , m_count(other.m_count)
        {
            other.m_buffer = nullptr;
            other.m_count = 0;
        }

        ~device_matrix()
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

        __host__ __device__ size_t GetCount() const
        {
            return m_count;
        }

        __host__ __device__ size_t GetSize() const
        {
            return m_count * sizeof(T);
        }

        /**
         * @brief Performs a synchronous copy of data into the device buffer
         * 
         * @param data 
         * @return __host__ 
         */
        __host__ void SetDataSync(const T* data)
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpy(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
            DebugCudaErrors();
        }

        __host__ void SetDataAsync(const T* data)
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpy(m_buffer, data, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
            DebugCudaErrors();
        }

        __host__ void ToHost(T* out) const
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpy(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }

        __host__ void ToHost(std::vector<T>& out) const
        {
            ToHost(out.data());
        } 

        __host__ void ToHostASync(T* out) const
        {
            size_t bytes = m_count * sizeof(T);
            checkCudaErrors(cudaMemcpyAsync(out, m_buffer, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
    };
}
}