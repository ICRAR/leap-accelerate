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

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <Eigen/Core>

#include <vector>

namespace icrar
{
    /**
     * Converts a casacore matrix to the equivalent eigen3 matrix
     */
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ToMatrix(const casacore::Matrix<T>& value)
    {
        auto shape = value.shape();
        auto output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(shape[0], shape[1]);
        std::copy(value.begin(), value.end(), output.reshaped().begin());
        return output;
    }

    /**
     * @brief Converts a casacore matrix to a fixed size eigen3 matrix
     * 
     * @tparam T scalar type
     * @tparam R rows
     * @tparam C columns
     * @param value casacore matrix to convert
     * @return Eigen::Matrix<T, R, C> 
     */
    template<typename T, int R, int C>
    Eigen::Matrix<T, R, C> ToFixedMatrix(const casacore::Matrix<T>& value)
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

    /**
     * @brief Converts an Eigen3 matrix to the equivalent casacore matrix
     */
    template<typename T, int R, int C>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, R, C>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(2, R, C), value.data());
    }

    /**
     * @brief Converts an Eigen3 matrix to the equivalent casacore matrix
     */
    template<typename T>
    casacore::Matrix<T> ConvertMatrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& value)
    {
        return casacore::Matrix<T>(casacore::IPosition(2, value.rows(), value.cols()), value.data());
    }

    /**
     * @brief Converts a casacore vector to the equivalent Eigen3 vector
     */
    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1> ToVector(casacore::Vector<T> value)
    {
        auto output = Eigen::Matrix<T, Eigen::Dynamic, 1>(value.size());
        std::copy(value.begin(), value.end(), output.reshaped().begin());
        return output;
    }

    /**
     * @brief Converts a std vector to the equivalent Eigen3 vector
     */
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
     * @tparam T scalar type
     * @param value eigen3 vector
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
     * @param value casacore uvw
     * @return icrar::MVuvw 
     */
    icrar::MVuvw ToUVW(const casacore::MVuvw& value);

    /**
     * @brief Converts a casacore UVW vector to an icrar UVW vector
     * 
     * @param value value to convert 
     * @return std::vector<icrar::MVuvw> 
     */
    std::vector<icrar::MVuvw> ToUVWVector(const std::vector<casacore::MVuvw>& value);
    
    /**
     * @brief Converts a column-major matrix of size Nx3 into a vector of UVWs
     * 
     * @param value value to convert
     * @return std::vector<icrar::MVuvw> 
     */
    std::vector<icrar::MVuvw> ToUVWVector(const Eigen::MatrixXd& value);

    /**
     * @brief Converts an icrar UVW value to a casacore UVW value
     */
    casacore::MVuvw ToCasaUVW(const icrar::MVuvw& value);

    /**
     * @brief Converts an icrar UVW vector to a casacore UVW vector
     */
    std::vector<casacore::MVuvw> ToCasaUVWVector(const std::vector<icrar::MVuvw>& value);

    /**
     * @brief Converts an icrar UVW vector to a casacore UVW vector
     * 
     * @param value value to convert
     * @return std::vector<casacore::MVuvw> 
     */
    std::vector<casacore::MVuvw> ToCasaUVWVector(const Eigen::MatrixX3d& value);

    /**
     * @brief Converts a casacore direction to an icrar spherical direction
     * 
     * @param value value to convert
     * @return SphericalDirection 
     */
    SphericalDirection ToDirection(const casacore::MVDirection& value);

    /**
     * @brief Converts a casacore Direction vector to an icrar Spherical Direction
     * 
     * @param value value to convert
     * @return std::vector<SphericalDirection> 
     */
    std::vector<SphericalDirection> ToDirectionVector(const std::vector<casacore::MVDirection>& value);

    /**
     * @brief Converts an icrar spherical direction to a casacore direction
     * 
     * @param value value to convert
     * @return casacore::MVDirection 
     */
    casacore::MVDirection ToCasaDirection(const SphericalDirection& value);

    /**
     * @brief Convers an icrar spherical direction vector to a casacore direction vector
     * 
     * @param value value to convert
     * @return std::vector<casacore::MVDirection> 
     */
    std::vector<casacore::MVDirection> ToCasaDirectionVector(const std::vector<SphericalDirection>& value);
} // namespace icrar