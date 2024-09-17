/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <gtest/gtest.h>

#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <array>

class MatrixTests : public testing::Test
{
    const double TOLERANCE = 0.0001;

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

    void TestTranspose()
    {
        auto m1 = Eigen::MatrixXd(3, 3);
        m1 <<
        1, 2, 3,
        4, 5, 6,
        7, 8, 9;
        
        auto m1t = m1.transpose();

        auto expected_m1t = Eigen::MatrixXd(3, 3);
        expected_m1t <<
        1, 4, 7,
        2, 5, 8,
        3, 6, 9;

        ASSERT_EQ(Eigen::MatrixXd(m1t), expected_m1t);
    }

    void TestSquareInvert()
    {
        auto m1 = Eigen::MatrixXd(3, 3);
        m1 <<
        3, 0, 2,
        2, 0, -2,
        0, 1, 1;
        
        auto m1d = m1.inverse();

        auto expected_m1d = Eigen::MatrixXd(3, 3);
        expected_m1d <<
        0.2, 0.2, 0,
        -0.2, 0.3, 1,
        0.2, -0.3, 0;
        
        ASSERT_MEQD(m1d, expected_m1d, TOLERANCE);
    }

    void TestPseudoInverse33()
    {
        auto m1 = Eigen::MatrixXd(3, 3);
        m1 <<
        3, 0, 2,
        2, 0, -2,
        0, 1, 1;

        auto m1d = icrar::cpu::pseudo_inverse(m1);

        auto expected_m1d = Eigen::MatrixXd(3, 3);
        expected_m1d <<
        0.2, 0.2, 0,
        -0.2, 0.3, 1,
        0.2, -0.3, 0;
        
        ASSERT_MEQD(expected_m1d, m1d, TOLERANCE);
    }

    void TestPseudoInverse32()
    {
        auto m1 = Eigen::MatrixXd(3, 2);
        m1 <<
        0.5, 0.5,
        -1, -1,
        -0.5, -0.5;

        auto m1d = icrar::cpu::pseudo_inverse(m1);

        auto expected_m1d = Eigen::MatrixXd(2, 3);
        expected_m1d <<
        0.166667, -0.333333, -0.166667,
        0.166667, -0.333333, -0.166667;

        ASSERT_MEQD(expected_m1d, m1d, TOLERANCE);
        ASSERT_MEQD(m1, m1 * m1d * m1, TOLERANCE);
    }

    void TestSVD42()
    {
        auto m1 = Eigen::MatrixXd(4, 2);
        m1 <<
        2, 4,
        1, 3,
        0, 0,
        0, 0;

        auto bdc = Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV>(m1);

        const auto& u = bdc.matrixU();
        const auto& v = bdc.matrixV();

        ASSERT_EQ(m1.rows(), u.rows());
        ASSERT_EQ(m1.rows(), u.cols());
        ASSERT_EQ(m1.cols(), v.rows());
        ASSERT_EQ(m1.cols(), v.cols());
    }

    void TestSVDPseudoInverse32Degenerate()
    {
        auto m1 = Eigen::MatrixXd(3, 2);
        m1 <<
        0.5, 0.5,
        -1, -1,
        -0.5, -0.5;

        auto m1d = icrar::cpu::SVDPseudoInverse(m1);

        auto expected_m1d = Eigen::MatrixXd(2, 3);
        expected_m1d <<
        0.166667, -0.333333, -0.166667,
        0.166667, -0.333333, -0.166667;

        ASSERT_MEQD(expected_m1d, m1d, TOLERANCE);
        ASSERT_MEQD(m1, m1 * m1d * m1, TOLERANCE);
    }

    void TestPseudoInverse42()
    {
        auto m1 = Eigen::MatrixXd(4, 2);
        m1 <<
        1, 2,
        3, 4,
        5, 6,
        7, 8;

        auto m1d = icrar::cpu::SVDPseudoInverse(m1);

        auto expected_m1d = Eigen::MatrixXd(2, 4);
        expected_m1d <<
        -1, -0.5, 0, 0.5,
        0.85, 0.45, 0.05, -0.35;

        ASSERT_MEQD(expected_m1d, m1d, TOLERANCE);
        ASSERT_MEQD(m1, m1 * m1d * m1, TOLERANCE);
        ASSERT_MEQD(Eigen::MatrixXd::Identity(2,2), m1d * m1, TOLERANCE);
    }
};

TEST_F(MatrixTests, TestMatrixSize) { TestMatrixSize(); }
TEST_F(MatrixTests, TestMatrixEigen) { TestMatrixEigen(); }
TEST_F(MatrixTests, TestMatrixMultiply) { TestMatrixMultiply(); }
TEST_F(MatrixTests, TestMatrixPretty) { TestMatrixPretty(); }

TEST_F(MatrixTests, TestTranspose) { TestTranspose(); }
TEST_F(MatrixTests, TestSquareInvert) { TestSquareInvert(); }
TEST_F(MatrixTests, TestPseudoInverse33) { TestPseudoInverse33(); }
TEST_F(MatrixTests, TestPseudoInverse32) { TestPseudoInverse32(); }
TEST_F(MatrixTests, TestSVD42) { TestSVD42(); }

TEST_F(MatrixTests, TestPseudoInverse42) { TestPseudoInverse42(); }
TEST_F(MatrixTests, TestSVDPseudoInverse32Degenerate) { TestSVDPseudoInverse32Degenerate(); }