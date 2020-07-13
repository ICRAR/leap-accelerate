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
    void test_array_add(bool useGpu)
    {
        std::array<int, n> a;
        std::array<int, n> b;
        std::array<int, n> c;

        a.fill(6);
        b.fill(10);

        if(useGpu)
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

    void test_vector_add(const int n, bool useGpu)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        if(useGpu)
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
