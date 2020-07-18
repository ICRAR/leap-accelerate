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

#include <icrar/leap-accelerate/math/cuda/matrix.h>

#include <icrar/leap-accelerate/math/cpu/svd.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> //Needed for Matrix::inverse()

#include <gtest/gtest.h>

#include <array>
#include <vector>
#include <sstream>

namespace
{
    void ASSERT_MEQ(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance)
    {
        ASSERT_EQ(expected.rows(), actual.rows());
        ASSERT_EQ(expected.cols(), actual.cols());
        if(!actual.isApprox(expected, tolerance))
        {
            std::cerr << "got\n" << actual << "\n" << " expected\n" << expected;
        }
        ASSERT_TRUE(actual.isApprox(expected, tolerance));
    }
}

class matrix_tests : public testing::Test
{
    const double TOLERANCE = 0.0001; 
public:
    matrix_tests()
    {

    }

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

    void test_svd()
    {
        auto m = Eigen::MatrixXd(7, 5);
        m <<
        1, 1, 1, 0, 0,
        2, 2, 2, 0, 0,
        1, 1, 1, 0, 0,
        5, 5, 5, 0, 0,
        0, 0, 0, 2, 2,
        0, 0, 0, 3, 3,
        0, 0, 0, 1, 1;

        Eigen::MatrixXd matU;
        Eigen::MatrixXd matV;
        std::tie(matU, matV) = icrar::cpu::SVD_slow(m);
        {
            auto expectedU = Eigen::MatrixXd(7, 5);
            expectedU <<
            0.179605, 7.90796e-17, 0.898027, 0.356143, 0.114959,
            0.359211, -1.96067e-16, 0.273465, -0.883008, -0.0668068,
            0.179605, 1.20406e-17, 0.136732, 0.0542259, -0.418932,
            0.898027, 6.02029e-17, -0.316338, 0.27113, 0.0875174,
            -0, 0.534522, -0, -0.093522, 0.755102,
            -0, 0.801784, -0, 0.077935, -0.462585,
            -0, 0.267261, -0, -0.046761, -0.122449;

            ASSERT_MEQ(matU, expectedU, TOLERANCE);
        }

        {
            auto expectedV = Eigen::MatrixXd(5, 5);
            expectedV <<
            0.57735, 0, -0.707107, 0, -0.408248,
            0.57735, 0, 0, 0, 0.816497,
            0.57735, 0, 0.707107, 0, -0.408248,
            0, 0.707107, 0, -0.707107, 0,
            0, 0.707107, 0, 0.707107, 0;

            ASSERT_MEQ(matV, expectedV, TOLERANCE);
        }

        //TODO: re-form original matrix
    }

    void test_transpose()
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

    void test_invert()
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
        
        ASSERT_MEQ(m1d, expected_m1d, TOLERANCE);
    }
};

TEST_F(matrix_tests, test_cpu_svd) { test_svd(); }
TEST_F(matrix_tests, test_transpose) { test_transpose(); }
TEST_F(matrix_tests, test_invert) { test_invert(); }