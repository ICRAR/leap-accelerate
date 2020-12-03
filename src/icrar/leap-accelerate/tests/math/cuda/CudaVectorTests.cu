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

class CudaVectorTests : public testing::Test
{
public:
    void SetUp() override
    {
        // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        ASSERT_EQ(1, deviceCount);
    }

    void TearDown() override
    {
        cudaDeviceReset();
    }

    template<typename T>
    void test_device_vector()
    {
        const int n = 10;
        auto out = std::vector<T>(n, 0);

        auto d_out = icrar::cuda::device_vector<T>(out);

        d_out.ToHost(out.data());
        std::vector<T> expected = std::vector<T>(n, 0);
        ASSERT_EQ(out, expected);
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

        std::vector<T> out = std::vector<T>(n, 0);
        icrar::cuda::h_add(a, b, out);

        std::vector<T> expected = std::vector<T>(n, 16);
        ASSERT_EQ(out, expected);
    }

    template<typename T>
    void test_device_vector_add(const int n)
    {
        auto a = std::vector<T>(n, 6);
        auto b = std::vector<T>(n, 10);
        auto c = std::vector<T>(n, 7);
        auto out = std::vector<T>(n, 0);

        auto d_a = icrar::cuda::device_vector<T>(a);
        auto d_b = icrar::cuda::device_vector<T>(b);
        auto d_c = icrar::cuda::device_vector<T>(c);
        auto d_out = icrar::cuda::device_vector<T>(out);

        icrar::cuda::h_add(d_a, d_b, d_out);
        icrar::cuda::h_add(d_c, d_out, d_out);

        d_out.ToHost(out.data());
        std::vector<T> expected = std::vector<T>(n, 23);
        ASSERT_EQ(out, expected);
    }
};

TEST_F(CudaVectorTests, test_device_vector) { test_device_vector<double>(); }

TEST_F(CudaVectorTests, test_gpu_array_add0) { test_array_add<1>(); }
TEST_F(CudaVectorTests, test_gpu_array_add3) { test_array_add<1000>(); }

TEST_F(CudaVectorTests, test_gpu_vector_add0) { test_vector_add<double>(1); }
TEST_F(CudaVectorTests, test_gpu_vector_add4) { test_vector_add<double>(10000); }
TEST_F(CudaVectorTests, test_gpu_vector_add6) { test_vector_add<double>(1000000); }

TEST_F(CudaVectorTests, test_gpu_device_vector_add0) { test_device_vector_add<double>(1); }
TEST_F(CudaVectorTests, test_gpu_device_vector_add4) { test_device_vector_add<double>(10000); }

