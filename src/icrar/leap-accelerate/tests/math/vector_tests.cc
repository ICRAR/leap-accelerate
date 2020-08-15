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

class vector_tests : public testing::Test
{
public:
    vector_tests()
    {

    }

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

    template<unsigned int n>
    void test_array_add(bool useCuda)
    {
        if(useCuda)
        {
            // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
            int deviceCount = 0;
            checkCudaErrors(cudaGetDeviceCount(&deviceCount));
            ASSERT_EQ(1, deviceCount);
        }

        std::array<int, n> a;
        std::array<int, n> b;
        std::array<int, n> c;

        a.fill(6);
        b.fill(10);

        if(useCuda)
        {
            icrar::cuda::add(n, a.data(), b.data(), c.data());
        }
        else
        {
            icrar::cpu::add(n, a.data(), b.data(), c.data());
        }

        std::array<int, n> expected;
        expected.fill(16);
        ASSERT_EQ(c, expected);
    }

    void test_vector_add(const int n, bool useCuda)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        if(useCuda)
        {
            icrar::cuda::add(a, b, c);
        }
        else
        {
            icrar::cpu::add(a, b, c);
        }

        std::vector<int> expected = std::vector<int>(n, 16);
        ASSERT_EQ(c, expected);
    }

    void test_device_vector_add(const int n, bool useCuda)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        if(useCuda)
        {
            auto d_a = icrar::cuda::device_vector<int>(a);
            auto d_b = icrar::cuda::device_vector<int>(b);
            auto d_c = icrar::cuda::device_vector<int>(c);
            icrar::cuda::add(d_a, d_b, d_c);
            d_c.ToHost(c);
        }
        else
        {
            icrar::cpu::add(a, b, c);
        }

        std::vector<int> expected = std::vector<int>(n, 16);
        ASSERT_EQ(c, expected);
    }

    void test_device_vector_fibonacci(const int n, const int k, bool useCuda)
    {
        std::vector<int> a = std::vector<int>(n, 1);
        std::vector<int> b = std::vector<int>(n, 1);
        std::vector<int> out = std::vector<int>(n, 0);

        if(useCuda)
        {
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

TEST_F(vector_tests, test_cpu_array_add0) { test_array_add<1>(false); }
TEST_F(vector_tests, test_cpu_array_add3) { test_array_add<1000>(false); }
TEST_F(vector_tests, test_cpu_vector_add0) { test_vector_add(1, false); }
TEST_F(vector_tests, test_cpu_vector_add4) { test_vector_add(10000, false); }
TEST_F(vector_tests, test_cpu_vector_add6) { test_vector_add(1000000, false); }

TEST_F(vector_tests, test_gpu_array_add0) { test_array_add<1>(true); }
TEST_F(vector_tests, test_gpu_array_add3) { test_array_add<1000>(true); }
TEST_F(vector_tests, test_gpu_vector_add0) { test_vector_add(1, true); }
TEST_F(vector_tests, test_gpu_vector_add4) { test_vector_add(10000, true); }
TEST_F(vector_tests, test_gpu_vector_add6) { test_vector_add(1000000, true); }

TEST_F(vector_tests, test_cpu_device_vector_add) { test_device_vector_add(1, false); }
TEST_F(vector_tests, test_gpu_device_vector_add) { test_device_vector_add(1, true); }

TEST_F(vector_tests, test_cpu_device_vector_fibonacci) { test_device_vector_fibonacci(100000, 20, false); }
TEST_F(vector_tests, test_gpu_device_vector_fibonacci) { test_device_vector_fibonacci(100000, 20, true); }

