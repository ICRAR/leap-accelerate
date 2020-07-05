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

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <eigen3/Eigen/Core>

namespace icrar
{
    Eigen::MatrixXd ConvertMatrix(casacore::Matrix<double> value)
    {
        auto shape = value.shape();
        auto m = Eigen::MatrixXd(shape[0], shape[1]);

        auto it = value.begin();
        for(int row = 0; row < shape[0]; ++row)
        {
            for(int col = 0; col < shape[1]; ++col)
            {
                m(row, col) = *it;
                it++;
            }
        }
        return m;
    }

    casacore::Matrix<double> ConvertMatrix(Eigen::MatrixXd value)
    {
        Eigen::MatrixXd m(value.rows(), value.cols());
        for(int row = 0; row < value.rows(); ++row)
        {
            for(int col = 0; col < value.cols(); ++col)
            {
                m(row, col) = 0;
            }
        }
        return casacore::Matrix<double>(casacore::IPosition(value.rows(), value.cols()), m.data());
    }


    Eigen::MatrixXcd ConvertMatrix(casacore::Matrix<std::complex<double>> v)
    {

    }

    Eigen::VectorXd ConvertVector(casacore::Array<double> value)
    {
        auto v = Eigen::VectorXd(value.size());
    }

    // Eigen::VectorXcd ConvertVector(casacore::CArray<double> v)
    // {

    // }
}