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
#include <vector>

namespace icrar
{
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ConvertMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        auto m = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(shape[0], shape[1]);

        auto it = value.begin();
        for(int col = 0; col < shape[1]; ++col)
        {
            for(int row = 0; row < shape[0]; ++row)
            {
                //eigen column major format
                m(row, col) = *it;
                it++;
            }
        }
        return m;
    }

    template<typename T, int R, int C>
    Eigen::Matrix<T, R, C> ConvertMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        if(shape[0] != R || shape[1] != C)
        {
            throw std::invalid_argument("matrix shape does not match template");
        }

        auto m = Eigen::Matrix<T, R, C>();

        auto it = value.begin();
        for(int col = 0; col < C; ++col)
        {
            for(int row = 0; row < R; ++row)
            {
                m(row, col) = *it;
                it++;
            }
        }
        return m;
    }

    template<typename T, int R, int C>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, R, C>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(std::vector<int>{R, C}), value.data());
    }

    template<typename T>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(std::vector<int>{(int)value.rows(), (int)value.cols()}), value.data());
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1> ConvertVector(casacore::Array<T> value)
    {
        auto v = Eigen::Matrix<T, Eigen::Dynamic, 1>(value.size());
        return v;
    }

    template<typename T>
    casacore::Array<T> ConvertVector(Eigen::Matrix<T, Eigen::Dynamic, 1> value)
    {
        auto v = casacore::Array<T>(casacore::IPosition(value.rows()), value.data());
        return v;
    }
}