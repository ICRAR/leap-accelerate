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

#pragma once

#include <icrar/leap-accelerate/model/cuda/DeviceMetaData.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <icrar/leap-accelerate/common/Tensor3X.h>

#include <gtest/gtest.h>

//complex double
void assert_eqcd(const std::complex<double>& expected, const std::complex<double>& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

//matrix equal int
void assert_meqi(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

//matrix equal double
void assert_meqd(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);
void assert_meq3d(const Eigen::Matrix3d& expected, const Eigen::Matrix3d& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

//matrix equal complex double
void assert_meqcd(const Eigen::MatrixXcd& expected, const Eigen::MatrixXcd& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

//vector equal int
void assert_veqi(const Eigen::VectorXi& expected, const Eigen::VectorXi& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

//vector equal double
void assert_veqd(const Eigen::VectorXd& expected, const Eigen::VectorXd& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

void assert_veqd(const std::vector<double>& expected, const std::vector<double>& actual, double tolerance, std::string ln, std::string rn, std::string file, int line);

template<typename T>
void assert_teq(const Eigen::Tensor<T, 3>& expected, const Eigen::Tensor<T, 3>& actual, double tolerance, std::string ln, std::string rn, std::string file, int line)
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

void assert_metadataeq(const icrar::cpu::MetaData& expected, const icrar::cpu::MetaData& actual, std::string ln, std::string rn, std::string file, int line);

#define ASSERT_EQCD(expected, actual, tolerance) assert_eqcd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

#define ASSERT_MEQI(expected, actual, tolerance) assert_meqi(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
#define ASSERT_MEQ(expected, actual, tolerance) assert_meqd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
#define ASSERT_MEQ3D(expected, actual, tolerance) assert_meqd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
#define ASSERT_MEQCD(expected, actual, tolerance) assert_meqcd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

#define ASSERT_VEQI(expected, actual, tolerance) assert_veqi(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)
#define ASSERT_VEQD(expected, actual, tolerance) assert_veqd(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

#define ASSERT_TEQ(expected, actual, tolerance) assert_teq(expected, actual, tolerance, #expected, #actual, __FILE__, __LINE__)

#define ASSERT_MDEQ(expected, actual, tolerance) assert_metadataeq(expected, actual, #expected, #actual, __FILE__, __LINE__)