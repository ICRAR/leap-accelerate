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

#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/math/cpu/vector.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>

#include <gtest/gtest.h>

#include <array>
#include <vector>

class VectorTests : public testing::Test
{
public:
    VectorTests() = default;

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

    template<unsigned int n>
    void TestCpuArrayAdd(bool useCuda)
    {
        if(useCuda)
        {
#ifdef CUDA_ENABLED
            // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
            int deviceCount = 0;
            checkCudaErrors(cudaGetDeviceCount(&deviceCount));
            ASSERT_EQ(1, deviceCount);
#endif
        }

        std::array<int, n> a = {};
        std::array<int, n> b = {};
        std::array<int, n> c = {};

        a.fill(6);
        b.fill(10);

        if(useCuda)
        {
#ifdef CUDA_ENABLED
            icrar::cuda::add(n, a.data(), b.data(), c.data());
#endif
        }
        else
        {
            icrar::cpu::add(n, a.data(), b.data(), c.data());
        }

        std::array<int, n> expected = {};
        expected.fill(16);
        ASSERT_EQ(c, expected);
    }

    void TestVectorAdd(const int n, bool useCuda)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        if(useCuda)
        {
#ifdef CUDA_ENABLED
            icrar::cuda::add(a, b, c);
#endif
        }
        else
        {
            icrar::cpu::add(a, b, c);
        }

        std::vector<int> expected = std::vector<int>(n, 16);
        ASSERT_EQ(c, expected);
    }

    void TestDeviceVectorAdd(const int n, bool useCuda)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        if(useCuda)
        {
#ifdef CUDA_ENABLED
            auto d_a = icrar::cuda::device_vector<int>(a);
            auto d_b = icrar::cuda::device_vector<int>(b);
            auto d_c = icrar::cuda::device_vector<int>(c);
            icrar::cuda::add(d_a, d_b, d_c);
            d_c.ToHost(c);
#endif
        }
        else
        {
            icrar::cpu::add(a, b, c);
        }

        std::vector<int> expected = std::vector<int>(n, 16);
        ASSERT_EQ(c, expected);
    }

    void TestDeviceVectorFibonacci(const int n, const int k, bool useCuda)
    {
        std::vector<int> a = std::vector<int>(n, 1);
        std::vector<int> b = std::vector<int>(n, 1);
        std::vector<int> out = std::vector<int>(n, 0);

        if(useCuda)
        {
#ifdef CUDA_ENABLED
            auto d_a = icrar::cuda::device_vector<int>(a);
            auto d_b = icrar::cuda::device_vector<int>(b);
            auto d_c = icrar::cuda::device_vector<int>(out);

            auto n1 = &d_a;
            auto n2 = &d_b;
            auto n3 = &d_c;
            for(int i = 0; i < k; i++)
            {
                icrar::cuda::add(*n1, *n2, *n3);
                n1 = n2;
                n2 = n3;
            }
            n3->ToHost(out);
#endif
        }
        else
        {
            auto n1 = &a;
            auto n2 = &b;
            auto n3 = &out;
            for(int i = 0; i < k; i++)
            {
                icrar::cpu::add(*n1, *n2, *n3);
                n1 = n2;
                n2 = n3;
            }
            out = *n3;
        }

        std::vector<int> expected = std::vector<int>(n, 786432);
        ASSERT_EQ(out, expected);
    }
};

TEST_F(VectorTests, TestCpuArrayAdd0) { TestCpuArrayAdd<1>(false); }
TEST_F(VectorTests, TestCpuArrayAdd3) { TestCpuArrayAdd<1000>(false); }
TEST_F(VectorTests, TestCpuVectorAdd3) { TestVectorAdd(1, false); }
TEST_F(VectorTests, TestCpuVectorAdd4) { TestVectorAdd(10000, false); }
TEST_F(VectorTests, TestCpuVectorAdd6) { TestVectorAdd(1000000, false); }
TEST_F(VectorTests, TestCpuDeviceVectorAdd) { TestDeviceVectorAdd(1, false); }
TEST_F(VectorTests, TestCpuDeviceVectorFibonacci) { TestDeviceVectorFibonacci(100000, 20, false); }

#if CUDA_ENABLED
TEST_F(VectorTests, TestGpuArrayAdd0) { TestCpuArrayAdd<1>(true); }
TEST_F(VectorTests, TestGpuArrayAdd3) { TestCpuArrayAdd<1000>(true); }
TEST_F(VectorTests, TestGpuVectorAdd3) { TestVectorAdd(1, true); }
TEST_F(VectorTests, TestGpuVectorAdd4) { TestVectorAdd(10000, true); }
TEST_F(VectorTests, TestGpuVectorAdd6) { TestVectorAdd(1000000, true); }
TEST_F(VectorTests, TestGpuDeviceVectorAdd) { TestDeviceVectorAdd(1, true); }
TEST_F(VectorTests, TestGpuDeviceVectorFibonacci) { TestDeviceVectorFibonacci(100000, 20, true); }
#endif
