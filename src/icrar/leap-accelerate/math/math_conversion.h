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

#include <icrar/leap-accelerate/common/MVuvw.h>
#include <icrar/leap-accelerate/common/MVDirection.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <Eigen/Core>

#include <vector>

namespace icrar
{
    /**
     * 
     */
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ToMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        auto output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(shape[0], shape[1]);
        std::copy(value.begin(), value.end(), output.reshaped().begin());
        return output;
    }

    template<typename T, int R, int C>
    Eigen::Matrix<T, R, C> ToMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        if(shape[0] != R || shape[1] != C)
        {
            throw std::invalid_argument("matrix shape does not match template");
        }

        auto output = Eigen::Matrix<T, R, C>();
        std::copy(value.begin(), value.end(), output.reshaped().begin());
        return output;
    }

    template<typename T, int R, int C>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, R, C>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(2, R, C), value.data());
    }

    template<typename T>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(2, (int)value.rows(), (int)value.cols()), value.data());
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1> ToVector(casacore::Vector<T> value)
    {
        auto output = Eigen::Matrix<T, Eigen::Dynamic, 1>(value.size());
        std::copy(value.begin(), value.end(), output.reshaped().begin());
        return output;
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1> ToVector(const std::vector<T>& value)
    {
        auto output = Eigen::Matrix<T, Eigen::Dynamic, 1>(value.size());
        std::copy(value.begin(), value.end(), output.reshaped().begin());
        return output;
    }

    /**
     * @brief Converts an Eigen3 column-vector into a casacore Array
     * 
     * @tparam T 
     * @param value 
     * @return casacore::Array<T> 
     */
    template<typename T>
    casacore::Vector<T> ConvertVector(const Eigen::Matrix<T, Eigen::Dynamic, 1>& value)
    {
        return casacore::Vector<T>(casacore::IPosition(1, value.rows()), value.data());
    }

    /**
     * @brief Converts a casacore UVW value to an icrar UVW value
     * 
     * @param value 
     * @return icrar::MVuvw 
     */
    icrar::MVuvw ToUVW(const casacore::MVuvw& value);

    std::vector<icrar::MVuvw> ToUVWVector(const std::vector<casacore::MVuvw>& value);
    
    /**
     * @brief Converts a column-major matrix of size Nx3 into a vector of UVWs
     * 
     * @param value 
     * @return std::vector<icrar::MVuvw> 
     */
    std::vector<icrar::MVuvw> ToUVWVector(const Eigen::MatrixXd& value);
    casacore::MVuvw ToCasaUVW(const icrar::MVuvw& value);
    std::vector<casacore::MVuvw> ToCasaUVWVector(const std::vector<icrar::MVuvw>& value);
    std::vector<casacore::MVuvw> ToCasaUVWVector(const Eigen::MatrixX3d& value);

    icrar::MVDirection ToDirection(const casacore::MVDirection& value);
    std::vector<icrar::MVDirection> ToDirectionVector(const std::vector<casacore::MVDirection>& value);

    casacore::MVDirection ToCasaDirection(const icrar::MVDirection& value);
    std::vector<casacore::MVDirection> ToCasaDirectionVector(const std::vector<icrar::MVDirection>& value);
}
