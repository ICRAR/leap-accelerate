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
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <iostream>
#include <array>

class CasacoreMatrixTests : public testing::Test
{
public:
    CasacoreMatrixTests()
    {

    }

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }

    void test_column_major()
    {
        int nrows = 4;
        int ncolumns = 2;
        casacore::Matrix<double> m = casacore::Matrix<double>(nrows, ncolumns);
        ASSERT_EQ(nrows, m.nrow());
        ASSERT_EQ(ncolumns, m.ncolumn());

        double inc = 0.0;
        for(double& v : m)
        {
            v = inc;
            inc += 1.0;
        }

        inc = 0.0;
        //column major iteration
        for(int col = 0; col < ncolumns; col++)
        {
            for(int row = 0; row < nrows; row++)
            {
                ASSERT_EQ(inc, m(row,col));
                inc++;
            }
        }

        ASSERT_EQ(0.0, m(0,0));
        ASSERT_EQ(1.0, m(1,0));
        ASSERT_EQ(2.0, m(2,0));
        ASSERT_EQ(3.0, m(3,0));

        ASSERT_EQ(4.0, m(0,1));
        ASSERT_EQ(5.0, m(1,1));
        ASSERT_EQ(6.0, m(2,1));
        ASSERT_EQ(7.0, m(3,1));
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
        double inc = 0.0;
        for(double& v : m)
        {
            v = inc;
            inc += 1.0;
        }
        
        //Convert to eigen matrix and back
        Eigen::Matrix<double, 5, 5> em = icrar::ToMatrix<double, 5, 5>(m);
        casacore::Matrix<double> cm = icrar::ConvertMatrix<double, 5, 5>(em);
        

        inc = 0.0;
        for(int col = 0; col < em.cols(); ++col)
        {
            for(int row = 0; row < em.rows(); ++row)
            {
                ASSERT_EQ(inc, m(row,col));
                ASSERT_EQ(inc, em(row,col));
                ASSERT_EQ(inc, cm(row,col));
                inc += 1.0;
            }
        }

        inc = 0.0;
        for(double& v : cm)
        {
            ASSERT_EQ(inc, v);
            inc += 1.0;
        }
    }

    void test_matrix_casa_to_eigen_dynamic()
    {
        //Convert to dynamic eigen matrix and back
        auto shape = casacore::IPosition(std::vector<int>{5,5});
        Eigen::MatrixXd exm(shape[0], shape[1]);
        double inc = 0.0;
        for(int col = 0; col < exm.cols(); ++col)
        {
            for(int row = 0; row < exm.rows(); ++row)
            {
                exm(row, col) = inc;
                inc += 1.0;
            }
        }

        casacore::Matrix<double> cm = icrar::ConvertMatrix<double>(exm);
        
        inc =  0.0;
        for(auto it = cm.begin(); it != cm.end(); ++it)
        {
            ASSERT_EQ(inc, *it);
            inc += 1.0;
        }
    }

    void test_uvw_to_icrar()
    {
        casacore::MVuvw casa = casacore::MVuvw(1,2,3);
        icrar::MVuvw uvw = icrar::ToUVW(casa);

        ASSERT_EQ(1, uvw(0));
        ASSERT_EQ(2, uvw(1));
        ASSERT_EQ(3, uvw(2));
    }
};

TEST_F(CasacoreMatrixTests, test_column_major) { test_column_major(); }
TEST_F(CasacoreMatrixTests, test_matrix_casa) { test_matrix_casa(); }
TEST_F(CasacoreMatrixTests, test_matrix_casa_to_eigen) { test_matrix_casa_to_eigen(); }
TEST_F(CasacoreMatrixTests, test_matrix_casa_to_eigen_dynamic) { test_matrix_casa_to_eigen_dynamic(); }

TEST_F(CasacoreMatrixTests, test_uvw_to_icrar) { test_uvw_to_icrar(); }
