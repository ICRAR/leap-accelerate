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

#include "eigen_helper.h"

#include <icrar/leap-accelerate/common/stream_extensions.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>

void assert_near_cd(const std::complex<double>& expected, const std::complex<double>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    std::cerr << std::setprecision(15);
    if(std::abs(expected.real() - actual.real()) > tolerance || std::abs(expected.imag() - actual.imag()) > tolerance)
    {
        std::cerr << file << ":" << line << " " << ln << "!=" << rn << "\n";
        std::cerr << "expected: " << expected.real() << " + " << expected.imag() << "i\n"; 
        std::cerr << "got: " << actual.real() << " + " << actual.imag() << "i\n"; 
    }
    EXPECT_NEAR(expected.real(), actual.real(), tolerance);
    EXPECT_NEAR(expected.imag(), actual.imag(), tolerance);
}

template<typename T>
void assert_near_matrix(const Eigen::Matrix<T, -1, -1>& expected, const Eigen::Matrix<T, -1, -1>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " Matrix elements differ at:\n";
        
        for(int col = 0; col < actual.cols(); ++col)
        {
            for(int row = 0; row < actual.rows(); ++row)
            {
                if(std::abs(expected(row, col) - actual(row, col)) > tolerance)
                {
                    std::cerr << "expected(" << row << ", " << col << ") == " << expected(row, col) << "\n";
                    std::cerr << "actual(" << row << ", " << col << ") == " << actual(row, col) << "\n";
                }
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_near_matrix_i(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    assert_near_matrix<int>(expected, actual, tolerance, ln, rn, file, line);
}

void assert_near_matrix_d(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    assert_near_matrix<double>(expected, actual, tolerance, ln, rn, file, line);
}

void assert_near_matrix3_d(const Eigen::Matrix3d& expected, const Eigen::Matrix3d& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " Matrix elements differ at:\n";
        
        for(int col = 0; col < actual.cols(); ++col)
        {
            for(int row = 0; row < actual.rows(); ++row)
            {
                if(abs(expected(row, col) - actual(row, col)) > tolerance)
                {
                    std::cerr << "expected(" << row << ", " << col << ") == " << expected(row, col) << "\n";
                    std::cerr << "actual(" << row << ", " << col << ") == " << actual(row, col) << "\n";
                }
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_near_matrix_cd(const Eigen::MatrixXcd& expected, const Eigen::MatrixXcd& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    assert_near_matrix<std::complex<double>>(expected, actual, tolerance, ln, rn, file, line);
}

template<typename T>
void assert_near_vector(const Eigen::Matrix<T, -1, 1>& expected, const Eigen::Matrix<T, -1, 1>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " Vector elements differ at:\n";
        
        for(int row = 0; row < actual.rows(); ++row)
        {
            if(abs(expected(row) - actual(row)) > tolerance)
            {
                std::cerr << "expected(" << row << ") == " << expected(row) << "\n";
                std::cerr << "actual(" << row << ") == " << actual(row) << "\n";
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_near_vector_i(const Eigen::VectorXi& expected, const Eigen::VectorXi& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    assert_near_vector<int>(expected, actual, tolerance, ln, rn, file, line);
}

void assert_near_vector_d(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    assert_near_vector<double>(expected, actual, tolerance, ln, rn, file, line);
}

template<typename T>
void assert_near_vector(const std::vector<T>& expected, const std::vector<T>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    ASSERT_EQ(expected.size(), actual.size());
    if(!icrar::isApprox(expected, actual, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " std::vector elements differ at:\n" << std::setprecision(15);
        
        for(size_t i = 0; i < actual.size(); ++i)
        {
            if(abs(expected[i] - actual[i]) > tolerance)
            {
                std::cerr << "expected[" << i << "] == " << expected[i] << "\n";
                std::cerr << "actual[" << i << "] == " << actual[i] << "\n";
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(icrar::isApprox(expected, actual, tolerance));
}

void assert_near_vector_d(const std::vector<double>& expected, const std::vector<double>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    assert_near_vector<double>(expected, actual, tolerance, ln, rn, file, line);
}
