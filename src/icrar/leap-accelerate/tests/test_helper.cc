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

#include "test_helper.h"

template<typename T>
void assert_meq(const Eigen::Matrix<T, -1, -1>& expected, const Eigen::Matrix<T, -1, -1>& actual, double tolerance, std::string file, int line)
{
    ASSERT_EQ(expected.rows(), actual.rows());
    ASSERT_EQ(expected.cols(), actual.cols());
    if(!actual.isApprox(expected, tolerance))
    {
        std::cerr << "got\n" << actual << "\n" << " expected\n" << expected << "\n" << "at " << file << ":" << line << std::endl;
    }
    ASSERT_TRUE(actual.isApprox(expected, tolerance));
}

void assert_meqi(const Eigen::MatrixXi& expected, const Eigen::MatrixXi& actual, double tolerance, std::string file, int line)
{
    assert_meq<int>(expected, actual, tolerance, file, line);
}

void assert_meqd(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& actual, double tolerance, std::string file, int line)
{
    assert_meq<double>(expected, actual, tolerance, file, line);
}

void assert_meqcd(const Eigen::MatrixXcd& expected, const Eigen::MatrixXcd& actual, double tolerance, std::string file, int line)
{
    assert_meq<std::complex<double>>(expected, actual, tolerance, file, line);
}