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
#pragma once

#include <icrar/leap-accelerate/config.h>
#include <icrar/leap-accelerate/math/complex_extensions.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <gtest/gtest.h>

//complex double
void assert_near_cd(const std::complex<double>& expected, const std::complex<double>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

//matrix equal int
void assert_near_matrix_i(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual, int tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

//matrix equal double
void assert_near_matrix_d(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);
void assert_near_matrix3_d(const Eigen::Matrix3d& expected, const Eigen::Matrix3d& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

//matrix equal complex double
void assert_near_matrix_cd(const Eigen::MatrixXcd& expected, const Eigen::MatrixXcd& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

//vector equal bool
void assert_near_vector_b(const Eigen::VectorXb& expected, const Eigen::VectorXb& actual, const std::string& ln, const std::string& rn, const std::string& file, int line);

//vector equal int
void assert_near_vector_i(const Eigen::VectorXi& expected, const Eigen::VectorXi& actual, int tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

//vector equal double
void assert_near_vector_d(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

void assert_near_vector_d(const std::vector<double>& expected, const std::vector<double>& actual, double tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line);

template<typename T>
void assert_near_tensor(const Eigen::Tensor<T, 3>& expected, const Eigen::Tensor<T, 3>& actual, T tolerance, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    ASSERT_EQ(expected.dimensions(), actual.dimensions());
    ASSERT_EQ(expected.dimension(0), actual.dimension(0));
    ASSERT_EQ(expected.dimension(1), actual.dimension(1));
    ASSERT_EQ(expected.dimension(2), actual.dimension(2));
    if(!icrar::isApprox(actual, expected, tolerance))
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << file << ":" << line << " Tensor elements differ at:\n";
        
        for(int x = 0; x < actual.dimension(0); ++x)
        {
            for(int y = 0; y < actual.dimension(1); ++y)
            {
                for(int z = 0; z < actual.dimension(2); ++z)
                {
                    if(abs(expected(x, y, z) - actual(x, y, z)) > tolerance)
                    {
                        std::cerr << "expected(" << x << ", " << y << ", " << z << ") == " << expected(x, y, z) << "\n";
                        std::cerr << "actual(" << x << ", " << y << ", " << z << ") == " << actual(x, y, z) << "\n";
                    }
                }
            }
        }
        std::cerr << std::endl;
    }
    ASSERT_TRUE(icrar::isApprox(actual, expected, tolerance));
}

//tensor equal double
//void assert_teqd(const Eigen::Tensor<double, 3>& expected, const Eigen::Tensor<double, 3>& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);
//void assert_teqcd(const Eigen::Tensor<std::complex<double>, 3>& expected, const Eigen::Tensor<std::complex<double>, 3>& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_EQCD(expected, actual, tolerance) assert_near_cd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_MEQI(expected, actual, tolerance) assert_near_matrix_i(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_MEQD(expected, actual, tolerance) assert_near_matrix_d(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_MEQ3D(expected, actual, tolerance) assert_near_matrix3_d(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_MEQCD(expected, actual, tolerance) assert_near_matrix_cd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_VEQB(expected, actual) assert_near_vector_b(expected, actual, #expected, #actual, __FILE__, __LINE__)
// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_VEQI(expected, actual, tolerance) assert_near_vector_i(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_VEQD(expected, actual, tolerance) assert_near_vector_d(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

// NOLINTNEXTLINE(ppcoreguidelines-macro-usage)
#define ASSERT_TEQ(expected, actual, tolerance) assert_near_tensor(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)