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

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

/**
 * @brief Provides stream operator for std::vector as
 * a json-like literal.
 * 
 * @tparam T streamable type
 * @param os output stream
 * @param v vector
 * @return std::ostream& 
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "{";
    for (size_t i = 0; i < v.size(); ++i)
    { 
        os << v[i]; 
        if (i != v.size() - 1)
        { 
            os << ", ";
        }
    }
    os << "}\n"; 
    return os;
}

namespace icrar
{
    /**
     * @brief returns a linear sequence of values from start at step sized
     * intervals to the stop value inclusive
     * 
     * @tparam IntType integer type
     * @param start start index
     * @param stop exclusive end inex
     * @param step increment between generated elements
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop, IntType step)
    {
        if (step == IntType(0))
        {
            throw std::invalid_argument("step must be non-zero");
        }

        std::vector<IntType> result;
        IntType i = start;
        while ((step > 0) ? (i < stop) : (i > stop))
        {
            result.push_back(i);
            i += step;
        }

        return result;
    }

    /**
     * @brief returns a linear sequence of values from start to stop
     * 
     * @tparam IntType integer type
     * @param start start index
     * @param stop exclusive end index
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop)
    {
        return range(start, stop, IntType(1));
    }

    /**
     * @brief returns a linear sequence of values from 0 to stop
     * 
     * @tparam IntType integer type
     * @param stop exclusive end index
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType stop)
    {
        return range(IntType(0), stop, IntType(1));
    }

    /**
     * @brief Returns true if all vector elements of @p lhs are within the
     * tolerance threshold to @p rhs
     * 
     * @tparam T numeric type
     * @param lhs left hand side
     * @param rhs  right hand side
     * @param tolerance tolerance threshold
     */
    template<typename T>
    bool isApprox(const std::vector<T>& lhs, const std::vector<T>& rhs, T tolerance)
    {
        if(lhs.size() != rhs.size())
        {
            return false;
        }
        for(size_t i = 0; i < lhs.size(); ++i)
        {
            if(std::abs(lhs[i] - rhs[i]) >= tolerance)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Performs a std::transform on a collection into a
     * newly allocated std::vector
     * 
     * @tparam Func function of signature return_type(const value_type&)
     * @tparam Seq iterable collection of type value_type
     * @param func transformation function
     * @param seq iterable collection to transform
     * @return std::vector<value_type>
     */
    template <typename Func, typename Seq>
    auto vector_map(Func func, const Seq& seq)
    {
        using value_type = typename Seq::value_type;
        using return_type = std::result_of_t<Func(const value_type&)>;
        static_assert(
            std::is_assignable<std::function<return_type(const value_type&)>, Func>::value,
            "func argument must have a functor of signature return_type(const value_type&)");

        std::vector<return_type> result;
        result.reserve(seq.size());
        std::transform(seq.cbegin(), seq.cend(), std::back_inserter(result), func);
        return result;
    }
} // namespace icrar
