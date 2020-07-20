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
#include <icrar/leap-accelerate/math/cpu/Invert.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> //Needed for Matrix::inverse()

#include <gtest/gtest.h>

#include <array>
#include <vector>
#include <sstream>

namespace
{
    void assert_meq(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, std::string file, int line)
    {
        ASSERT_EQ(expected.rows(), actual.rows());
        ASSERT_EQ(expected.cols(), actual.cols());
        if(!actual.isApprox(expected, tolerance))
        {
            std::cerr << "got\n" << actual << "\n" << " expected\n" << expected << "\n" << "at " << file << ":" << line << std::endl;
        }
        ASSERT_TRUE(actual.isApprox(expected, tolerance));
    }
}

#define ASSERT_MEQ(expected, actual, tolerance) assert_meq(expected, actual, tolerance, __FILE__, __LINE__)

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

    void test_square_invert()
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

    void test_pseudo_inverse_33()
    {
        auto m1 = Eigen::MatrixXd(3, 3);
        m1 <<
        3, 0, 2,
        2, 0, -2,
        0, 1, 1;

        auto m1d = icrar::cpu::PseudoInverse(m1);

        auto expected_m1d = Eigen::MatrixXd(3, 3);
        expected_m1d <<
        0.2, 0.2, 0,
        -0.2, 0.3, 1,
        0.2, -0.3, 0;
        
        ASSERT_MEQ(expected_m1d, m1d, TOLERANCE);
    }

    void test_pseudo_inverse_32()
    {
        auto m1 = Eigen::MatrixXd(3, 2);
        m1 <<
        0.5, 0.5,
        -1, -1,
        -0.5, -0.5;

        auto m1d = icrar::cpu::PseudoInverse(m1);

        auto expected_m1d = Eigen::MatrixXd(2, 3);
        expected_m1d <<
        0.166667, -0.333333, -0.166667,
        0.166667, -0.333333, -0.166667;

        ASSERT_MEQ(expected_m1d, m1d, TOLERANCE);
        ASSERT_MEQ(m1, m1 * m1d * m1, TOLERANCE);
    }

    void test_svd33()
    {
        auto m1 = Eigen::MatrixXd(3, 3);
        m1 <<
        3, 0, 2,
        2, 0, -2,
        0, 1, 1;

        Eigen::MatrixXd u;
        Eigen::VectorXd s;
        Eigen::MatrixXd v;
        std::tie(u, s, v) = icrar::cpu::SVD(m1);

        // M = U * Sigma * Vt
        auto sig = Eigen::DiagonalMatrix<double, -1>(s);
        ASSERT_MEQ(m1, u * sig * v.transpose(), TOLERANCE);
    }

    void test_svd42_min()
    {
        auto m1 = Eigen::MatrixXd(4, 2);
        m1 <<
        2, 4,
        1, 3,
        0, 0,
        0, 0;

        auto bdc = Eigen::BDCSVD<Eigen::MatrixXd>(m1, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::MatrixXd u = bdc.matrixU();
        Eigen::MatrixXd v = bdc.matrixV();

        ASSERT_EQ(m1.rows(), u.rows());
        ASSERT_EQ(m1.rows(), u.cols());
        ASSERT_EQ(m1.cols(), v.rows());
        ASSERT_EQ(m1.cols(), v.cols());
    }

    void test_svd32()
    {
        auto m1 = Eigen::MatrixXd(3, 2);
        m1 <<
        0.5, 0.5,
        -1, -1,
        -0.5, -0.5;

        Eigen::MatrixXd u;
        Eigen::MatrixXd s;
        Eigen::MatrixXd v;
        std::tie(u, s, v) = icrar::cpu::SVDSigma(m1);

        // M = U * Sigma * Vt
        ASSERT_EQ(m1.rows(), u.rows());
        ASSERT_EQ(m1.rows(), u.cols());
        ASSERT_EQ(m1.rows(), s.rows());
        ASSERT_EQ(m1.cols(), s.cols());
        ASSERT_EQ(m1.cols(), v.rows());
        ASSERT_EQ(m1.cols(), v.cols());

        ASSERT_MEQ(m1, u * s * v.transpose(), TOLERANCE);
    }

    void test_right_invert()
    {
        auto m1 = Eigen::MatrixXd(3, 2);
        m1 <<
        0.5, 0.5,
        -1, -1,
        -0.5, -0.5;

        auto m1d = icrar::cpu::RightInvert(m1);

        auto expected_m1d = Eigen::MatrixXd(2, 3);
        expected_m1d <<
        0.166667, -0.333333, -0.166667,
        0.166667, -0.333333, -0.166667;

        ASSERT_MEQ(expected_m1d, m1d, TOLERANCE);
        ASSERT_MEQ(m1, m1 * m1d * m1, TOLERANCE);
    }
};

TEST_F(matrix_tests, test_transpose) { test_transpose(); }
TEST_F(matrix_tests, test_square_invert) { test_square_invert(); }
TEST_F(matrix_tests, test_pseudo_inverse_33) { test_pseudo_inverse_33(); }
TEST_F(matrix_tests, test_pseudo_inverse_32) { test_pseudo_inverse_32(); }
TEST_F(matrix_tests, test_svd33) { test_svd33(); }
TEST_F(matrix_tests, test_svd42_min) { test_svd42_min(); }
TEST_F(matrix_tests, test_svd32) { test_svd32(); }
//TEST_F(matrix_tests, test_right_invert) { test_right_invert(); }
