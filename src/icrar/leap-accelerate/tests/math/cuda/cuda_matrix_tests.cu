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
#include <icrar/leap-accelerate/math/cuda/matrix.cuh>
#include <icrar/leap-accelerate/math/cuda/vector.cuh>

#include <gtest/gtest.h>

#include <stdio.h>
#include <array>

class cuda_matrix_tests : public testing::Test
{
public:
    cuda_matrix_tests()
    {

    }

    void SetUp() override
    {
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        ASSERT_EQ(1, deviceCount);
        // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    }

    void TearDown() override
    {

    }

    template<typename T>
    void test_matrix_add()
    {
        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        auto a = MatrixXT(3,3);
        a << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

        auto b = a;
        auto c = MatrixXT(3,3); 

        icrar::cuda::h_add<T, -1, -1>(a, b, c);

        MatrixXT expected = a + b;
        ASSERT_EQ(c, expected);
    }

    template<typename T>
    void test_matrix_matrix_multiply()
    {
        using MatrixXT = Eigen::Matrix<T, -1, -1>;

        auto a = MatrixXT(3,3);
        a << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;

        auto b = MatrixXT(3,3);
        b << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;

        auto c = MatrixXT(3,3); 

        icrar::cuda::h_multiply(a, b, c);

        MatrixXT expected = a * b;

        //ASSERT_EQ(c, expected);
        ASSERT_EQ(c(0,0), 1);
        ASSERT_EQ(c(0,1), 0);
        ASSERT_EQ(c(0,2), 0);
        ASSERT_EQ(c(1,0), 0);
        ASSERT_EQ(c(1,1), 1);
        ASSERT_EQ(c(1,2), 0);
        ASSERT_EQ(c(2,0), 0);
        ASSERT_EQ(c(2,1), 0);
        ASSERT_EQ(c(2,2), 1);
    }

    template<typename T>
    void test_matrix_vector_multiply()
    {
        using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        auto a = MatrixXT(3,3);
        a << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

        auto b = Eigen::Matrix<T, Eigen::Dynamic, 1>(3, 1);
        auto c = Eigen::Matrix<T, Eigen::Dynamic, 1>(3, 1); 

        icrar::cuda::h_multiply(a, b, c);

        MatrixXT expected = a * b;
        ASSERT_EQ(c, expected);
    }

    template<typename T>
    void test_scalear_matrix_multiply()
    {

    }
};

TEST_F(cuda_matrix_tests, test_matrix_add) { test_matrix_add<double>(); }
TEST_F(cuda_matrix_tests, test_matrix_matrix_multiply) { test_matrix_matrix_multiply<double>(); }
TEST_F(cuda_matrix_tests, test_matrix_vector_multiply) { test_matrix_vector_multiply<double>(); }
TEST_F(cuda_matrix_tests, test_scalear_matrix_multiply) { test_scalear_matrix_multiply<double>(); }
