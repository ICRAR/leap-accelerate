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

#include <icrar/leap-accelerate/common/eigen_extensions.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <array>

class MatrixTests : public testing::Test
{
public:
    MatrixTests() = default;

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

    void TestMatrixSize()
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

    void TestMatrixEigen()
    {
        Eigen::Matrix<double, 3, 3> matrix;
    }

    void TestMatrixMultiply()
    {
        Eigen::Matrix<double, 3, 3> m1, m2, m3;
        m1 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
        m2 << 1, 0, 0, 0, 1, 0, 0, 0, 1;

        m3 = m1 * m2;

        Eigen::Matrix3d expected;
        expected << 1, 0, 0, 0, 1, 0, 0, 0, 1;

        ASSERT_EQ(expected, m3);
    }

    void TestMatrixPretty()
    {
        auto mat = Eigen::MatrixXd(5,5);
        mat <<
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24;

        mat = Eigen::MatrixXd(10,10);
        mat <<
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99;
    }
};

TEST_F(MatrixTests, TestMatrixSize) { TestMatrixSize(); }
TEST_F(MatrixTests, TestMatrixEigen) { TestMatrixEigen(); }
TEST_F(MatrixTests, TestMatrixMultiply) { TestMatrixMultiply(); }
TEST_F(MatrixTests, TestMatrixPretty) { TestMatrixPretty(); }
