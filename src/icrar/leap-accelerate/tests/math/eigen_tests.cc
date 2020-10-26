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

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <array>

class eigen_tests : public testing::Test
{
public:
    eigen_tests()
    {

    }

    void SetUp() override
    {
        //int deviceCount = 0;
        //cudaGetDeviceCount(&deviceCount);
        //ASSERT_NE(deviceCount, 0);

        // See this page: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    }

    void TearDown() override
    {

    }

    void test_matrix_size()
    {
        Eigen::Matrix<double, 1, 3> m13;
        ASSERT_EQ(3 * 8, sizeof(m13));

        Eigen::Matrix<double, 3, 1> m31;
        ASSERT_EQ(3 * 8, sizeof(m31));

        Eigen::Matrix<double, 3, 3> m33;
        ASSERT_EQ(9 * 8, sizeof(m33));

        //dynamically sized matrix uses a pointer
        //Eigen::Matrix<double, -1, -1> m33d(3,3);
        //ASSERT_EQ(9 * 8, sizeof(m33d));
    }

    void test_matrix_eigen()
    {
        Eigen::Matrix<double, 3, 3> matrix;
    }

    void test_matrix_multiply()
    {
        Eigen::Matrix<double, 3, 3> m1, m2, m3;
        m1 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
        m2 << 1, 0, 0, 0, 1, 0, 0, 0, 1;

        m3 = m1 * m2;

        Eigen::Matrix3d expected;
        expected << 1, 0, 0, 0, 1, 0, 0, 0, 1;

        ASSERT_EQ(expected, m3);
    }
};

TEST_F(eigen_tests, test_matrix_size) { test_matrix_size(); }
TEST_F(eigen_tests, test_matrix_eigen) { test_matrix_eigen(); }
TEST_F(eigen_tests, test_matrix_multiply) { test_matrix_multiply(); }
