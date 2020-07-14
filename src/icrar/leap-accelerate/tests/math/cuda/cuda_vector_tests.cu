/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#include <cuda_runtime.h>

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/math/cuda/vector.cuh>

#include <icrar/leap-accelerate/cuda/device_vector.h>

#include <gtest/gtest.h>

#include <stdio.h>
#include <array>

class cuda_vector_tests : public testing::Test
{
public:
    cuda_vector_tests()
    {

    }

    void SetUp() override
    {
        // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        ASSERT_EQ(1, deviceCount);
    }

    void TearDown() override
    {

    }

    template<int n>
    void test_array_add()
    {
        std::array<int, n> a;
        std::array<int, n> b;
        std::array<int, n> c;

        a.fill(6);
        b.fill(10);

        icrar::cuda::h_add<int, n>(a, b, c);

        std::array<int, n> expected;
        expected.fill(16);
        ASSERT_EQ(c, expected);
    }

    template<typename T>
    void test_vector_add(const int n)
    {
        std::vector<T> a = std::vector<T>(n, 6);
        std::vector<T> b = std::vector<T>(n, 10);
        std::vector<T> c = std::vector<T>(n, 0);

        icrar::cuda::h_add(a, b, c);

        std::vector<T> expected = std::vector<T>(n, 16);
        ASSERT_EQ(c, expected);
    }

    template<typename T>
    void multi_add(size_t n, const T* a, const T* b, T* c)
    {
        // total thread count may be greater than required
        constexpr int threadsPerBlock = 1024;
        int gridSize = (int)ceil((float)n / threadsPerBlock);
        size_t byteSize = n * sizeof(T);

        T* d_a;
        T* d_b;
        T* d_c;
        cudaMalloc((void**)&d_a, byteSize);
        cudaMalloc((void**)&d_b, byteSize);
        cudaMalloc((void**)&d_c, byteSize);

        checkCudaErrors(cudaMemcpy(d_a, a, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_b, b, byteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
        
        g_add<<<gridSize, threadsPerBlock>>>(n, d_a, d_b, d_c);
        g_add<<<gridSize, threadsPerBlock>>>(n, d_a, d_c, d_b);
        g_add<<<gridSize, threadsPerBlock>>>(n, d_a, d_b, d_c);

        cudaDeviceSynchronize();

        //cudaMemcpytoSymbol()
        checkCudaErrors(cudaMemcpy(c, d_c, byteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaFree(d_a));
        checkCudaErrors(cudaFree(d_b));
        checkCudaErrors(cudaFree(d_c));
    }
};

TEST_F(cuda_vector_tests, test_gpu_array_add0) { test_array_add<1>(); }
TEST_F(cuda_vector_tests, test_gpu_array_add3) { test_array_add<1000>(); }
TEST_F(cuda_vector_tests, test_gpu_vector_add0) { test_vector_add<double>(1); }
TEST_F(cuda_vector_tests, test_gpu_vector_add4) { test_vector_add<double>(10000); }
TEST_F(cuda_vector_tests, test_gpu_vector_add6) { test_vector_add<double>(1000000); }
