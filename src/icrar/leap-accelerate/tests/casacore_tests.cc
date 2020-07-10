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

#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/eigen_helper.h>
#include <icrar/leap-accelerate/utils.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>

#include <iostream>
#include <array>

class casacore_tests : public testing::Test
{
public:
    casacore_tests()
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

    void test_matrix_casa()
    {
        casacore::Matrix<double> m(5,5);
        ASSERT_EQ(25, m.size());

        // memory tends to be zerod but not always!
        icrar::ArrayFill(m, 0.0);
        for(auto it = m.begin(); it != m.end(); it++)
        {
            ASSERT_EQ(0.0, *it);
        }
        icrar::ArrayFill(m, 1.0);
        for(auto it = m.begin(); it != m.end(); it++)
        {
            ASSERT_EQ(1.0, *it);
        }
    }

    void test_matrix_casa_to_eigen()
    {
        casacore::Matrix<double> m(5,5);
        icrar::ArrayFill(m, 1.0);

        //Convert to eigen matrix and back
        Eigen::Matrix<double,5,5> em(m.data());
        //em(0,0) = 0;
        auto shape = casacore::IPosition(std::vector<int>({5,5}));
        casacore::Matrix<double> cm(shape, em.data());
        for(auto it = cm.begin(); it != cm.end(); ++it)
        {
            ASSERT_EQ(1.0, *it);
        }
    }

    void test_matrix_casa_to_eigen_dynamic()
    {
        //Convert to dynamic eigen matrix and back
        auto shape = casacore::IPosition(5,5);
        Eigen::MatrixXd exm(shape[0], shape[1]);
        for(int row = 0; row < exm.rows(); ++row)
        {
            for(int col = 0; col < exm.cols(); ++col)
            {
                exm(row, col) = 1;
            }
        }

        casacore::Matrix<double> cm(shape, exm.data());
        for(auto it = cm.begin(); it != cm.end(); ++it)
        {
            ASSERT_EQ(1.0, *it);
        }

        auto em = icrar::ConvertMatrix(cm);
        for(int row = 0; row < em.rows(); ++row)
        {
            for(int col = 0; col < em.cols(); ++col)
            {
                ASSERT_EQ(1.0, em(row,col));
            }
        }
    }
};

TEST_F(casacore_tests, test_matrix_casa) { test_matrix_casa(); }
TEST_F(casacore_tests, test_matrix_casa_to_eigen) { test_matrix_casa_to_eigen(); }
TEST_F(casacore_tests, test_matrix_casa_to_eigen_dynamic) { test_matrix_casa_to_eigen_dynamic(); }
