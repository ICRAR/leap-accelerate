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

#include <icrar/leap-accelerate/config.h>

#include <icrar/leap-accelerate/math/cpu/vector.h>

#ifdef USE_CUDA
#include <icrar/leap-accelerate/math/cuda/vector.h>
#include <icrar/leap-accelerate/cuda/device_vector.h>
#endif // USE_CUDA

#include <gtest/gtest.h>

#include <array>
#include <vector>


/// Used for function overloading
struct cpu {};
struct cuda {};


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

    template<typename ... Args>
    void add(cpu, Args && ... args)
    {
        icrar::cpu::add(std::forward<Args>(args)...);
    }

#ifdef USE_CUDA
    template<typename ... Args>
    void add(cuda, Args && ... args)
    {
        icrar::cuda::add(std::forward<Args>(args)...);
    }
#endif // USE_CUDA


    template<unsigned int n, typename backend>
    void test_array_add()
    {
        std::array<int, n> a;
        std::array<int, n> b;
        std::array<int, n> c;

        a.fill(6);
        b.fill(10);

        add(backend{}, n, a.data(), b.data(), c.data());

        std::array<int, n> expected;
        expected.fill(16);
        ASSERT_EQ(c, expected);
    }

    template<typename backend>
    void test_vector_add(const int n)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        add(backend{}, a, b, c);

        std::vector<int> expected = std::vector<int>(n, 16);
        ASSERT_EQ(c, expected);
    }

#ifdef USE_CUDA
    void test_device_vector_add(const int n)
    {
        std::vector<int> a = std::vector<int>(n, 6);
        std::vector<int> b = std::vector<int>(n, 10);
        std::vector<int> c = std::vector<int>(n, 2);

        auto d_a = icrar::cuda::device_vector<int>(a);
        auto d_b = icrar::cuda::device_vector<int>(b);
        auto d_c = icrar::cuda::device_vector<int>(c);
        icrar::cuda::add(d_a, d_b, d_c);
        d_c.ToHost(c);

        std::vector<int> expected = std::vector<int>(n, 16);
        ASSERT_EQ(c, expected);
    }
#endif // USE_CUDA

    template<typename backend, typename Container>
    void _fibonacci(const Container &a, const Container &b, Container &c, int k)
    {
        auto n1 = &a;
        auto n2 = &b;
        auto n3 = &c;
        for(int i = 0; i < k; i++)
        {
            add(backend{}, *n1, *n2, *n3);
            n1 = n2;
            n2 = n3;
        }
    }

    void fibonacci(cpu, const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out, int k)
    {
        _fibonacci<cpu>(a, b, out, k);
    }

#ifdef USE_CUDA
    void fibonacci(cuda, const std::vector<int> &a, const std::vector<int> &b, std::vector<int> &out, int k)
    {
        auto d_a = icrar::cuda::device_vector<int>(a);
        auto d_b = icrar::cuda::device_vector<int>(b);
        auto d_c = icrar::cuda::device_vector<int>(out);
        _fibonacci<cuda>(d_a, d_b, d_c, k);
        d_c.ToHost(out);
    }
#endif // USE_CUDA

    template<typename backend>
    void test_device_vector_fibonacci(const int n, const int k)
    {
        std::vector<int> a = std::vector<int>(n, 1);
        std::vector<int> b = std::vector<int>(n, 1);
        std::vector<int> out = std::vector<int>(n, 0);
        fibonacci(backend{}, a, b, out, k);
        std::vector<int> expected = std::vector<int>(n, 786432);
        ASSERT_EQ(out, expected);
    }
};

TEST_F(vector_tests, test_cpu_array_add0) { test_array_add<1, cpu>(); }
TEST_F(vector_tests, test_cpu_array_add3) { test_array_add<1000, cpu>(); }
TEST_F(vector_tests, test_cpu_vector_add0) { test_vector_add<cpu>(1); }
TEST_F(vector_tests, test_cpu_vector_add4) { test_vector_add<cpu>(10000); }
TEST_F(vector_tests, test_cpu_vector_add6) { test_vector_add<cpu>(1000000); }

TEST_F(vector_tests, test_cpu_device_vector_fibonacci) { test_device_vector_fibonacci<cpu>(100000, 20); }

#ifdef USE_CUDA
TEST_F(vector_tests, test_gpu_array_add0) { test_array_add<1, cuda>(); }
TEST_F(vector_tests, test_gpu_array_add3) { test_array_add<1000, cuda>(); }
TEST_F(vector_tests, test_gpu_vector_add0) { test_vector_add<cuda>(1); }
TEST_F(vector_tests, test_gpu_vector_add4) { test_vector_add<cuda>(10000); }
TEST_F(vector_tests, test_gpu_vector_add6) { test_vector_add<cuda>(1000000); }

TEST_F(vector_tests, test_gpu_device_vector_add) { test_device_vector_add(1); }

TEST_F(vector_tests, test_gpu_device_vector_fibonacci) { test_device_vector_fibonacci<cuda>(100000, 20); }
#endif // USE_CUDA
