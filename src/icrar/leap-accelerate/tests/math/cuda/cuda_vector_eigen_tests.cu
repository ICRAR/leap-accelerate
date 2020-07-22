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
#include <icrar/leap-accelerate/math/cuda/vector_eigen.cuh>

#ifdef __CUDACC_VER__
#undef __CUDACC_VER__
#define __CUDACC_VER__ ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif
#include <eigen3/Eigen/Core>

#include <gtest/gtest.h>

#include <stdio.h>
#include <array>

class cuda_vector_eigen_tests : public testing::Test
{
public:
    cuda_vector_eigen_tests()
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

    void test_vector_add()
    {
        constexpr int N = 10;
        auto a = Eigen::Matrix<double, N, 1>();
        a << 6,6,6,6,6, 6,6,6,6,6;

        auto b = Eigen::Matrix<double, N, 1>();
        b << 10,10,10,10,10, 10,10,10,10,10;

        auto c = Eigen::Matrix<double, N, 1>();

        icrar::cuda::h_add<double, N>(a, b, c);

        auto expected = Eigen::Matrix<double, N, 1>();
        expected << 16,16,16,16,16, 16,16,16,16,16;
        ASSERT_EQ(c, expected);
    }
};

TEST_F(cuda_vector_eigen_tests, test_gpu_vector_add0) { test_vector_add(); }
