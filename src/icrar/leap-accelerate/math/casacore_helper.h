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

#include <icrar/leap-accelerate/math/math_conversion.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <boost/optional.hpp>

#include <functional>
#include <algorithm>
#include <complex>
#include <type_traits>

namespace icrar
{
    template<typename T>
    casacore::Matrix<T> Transpose(const casacore::Matrix<T> matrix)
    {
        auto m = ConvertMatrix(matrix);
        return ConvertMatrix(m.Transpose());
    }

    template<typename T>
    icrar::MVuvw Dot(const icrar::MVuvw& left, const Eigen::Matrix<T, 3, 3>& right)
    {
         return left * right;
    }

    template<typename T>
    casacore::MVuvw Dot(const casacore::MVuvw& v1, const casacore::Matrix<T>& v2)
    {
        auto uvw = ToUVW(v1);
        auto mat = ToMatrix<T, 3, 3>(v2);
        return ToCasaUVW(Dot(uvw, mat));
    }

    template<typename T>
    bool Equal(const boost::optional<casacore::Matrix<T>>& l, const boost::optional<casacore::Matrix<T>>& r)
    {
        bool equal = false;
        if(l.is_initialized() && r.is_initialized())
        {
            if(l.value().shape() == r.value().shape())
            {
                equal = std::equal(l.value().cbegin(), l.value().cend(), r.value().cbegin());
            }
        }
        return equal;
    }

    template<typename T> 
    bool Equal(const casacore::Array<T>& l, const casacore::Array<T>& r)
    {
        bool equal = l.shape() == r.shape();
        if(equal)
        {
            equal = std::equal(l.cbegin(), l.cend(), r.cbegin());
        }
        return equal;
    }

    /**
     * @brief Performs a std::transform on a newly allocated casacore::Matrix
     * 
     * @tparam T The input vector template type
     * @tparam function of signature R(const T&)
     * @param vector 
     * @return std::vector<R> 
     */
    template<typename T, typename Op>
    casacore::Matrix<std::result_of_t<Op(const T&)>> casa_matrix_map(const casacore::Matrix<T>& matrix, Op lambda)
    {
        using R = std::result_of_t<Op(const T&)>;
        static_assert(std::is_assignable<std::function<R(const T&)>, Op>::value, "lambda argument must be a function of signature R(const T&)");

        auto result = casacore::Matrix<R>(matrix.shape());
        std::transform(matrix.cbegin(), matrix.cend(), result.begin(), lambda);
        return result;
    }

    /**
     * @brief Performs a std::transform on a newly allocated casacore::Vector
     * 
     * @tparam T The input vector template type
     * @tparam function of signature R(const T&)
     * @param vector 
     * @return std::vector<R> 
     */
    template<typename T, typename Op>
    casacore::Vector<std::result_of_t<Op(const T&)>> casa_vector_map(const casacore::Vector<T>& vector, Op lambda)
    {
        using R = std::result_of_t<Op(const T&)>;
        static_assert(std::is_assignable<std::function<R(const T&)>, Op>::value, "lambda argument must be a function of signature R(const T&)");

        auto result = casacore::Vector<R>(vector.shape());
        std::transform(vector.cbegin(), vector.cend(), result.begin(), lambda);
        return result;
    }
    
    template<typename T>
    void ArrayFill(casacore::Array<T>& value, T v)
    {
        for(auto it = value.begin(); it != value.end(); it++)
        {
            *it = v;
        }
    }

    /**
     * @brief Returns the largest value within the array
     * 
     * @tparam T 
     * @param value 
     * @return T 
     */
    template<typename T>
    T ArrayMax(const casacore::Array<T>& value)
    {
        T max = 0;
        for(auto it = value.begin(); it != value.end(); it++)
        {
            max = std::max(max, *it);
        }
        return max;
    }
}