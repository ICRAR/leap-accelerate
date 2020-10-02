
/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <type_traits>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "{";
    for (int i = 0; i < v.size(); ++i)
    { 
        os << v[i]; 
        if (i != v.size() - 1) 
            os << ", "; 
    } 
    os << "}\n"; 
    return os;
}

namespace icrar
{
    /**
     * @brief Returns of true if all vector elements of @param lhs are within the threshold difference to @param rhs 
     * 
     * @tparam T 
     * @param lhs 
     * @param rhs 
     * @param threshold 
     * @return true 
     * @return false 
     */
    template<typename T>
    bool isApprox(const std::vector<T>& lhs, const std::vector<T>& rhs, T threshold)
    {
        if(lhs.size() != rhs.size())
        {
            return false;
        }
        
        for(size_t i = 0; i < lhs.size(); ++i)
        {
            if(std::abs(lhs[i] - rhs[i]) >= threshold)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Performs a std::transform on a newly allocated std::vector 
     * 
     * @tparam T The input vector template type
     * @tparam function of signature R(const T&)
     * @param vector 
     * @return std::vector<R> 
     */
    template<typename T, typename Op>
    std::vector<std::result_of_t<Op(const T&)>> vector_map(const std::vector<T>& vector, Op lambda)
    {
        using R = std::result_of_t<Op(const T&)>;
        static_assert(std::is_assignable<std::function<R(const T&)>, Op>::value, "lambda argument must be a function of signature R(const T&)");

        auto result = std::vector<R>();
        result.reserve(vector.size());
        std::transform(vector.cbegin(), vector.cend(), std::back_inserter(result), lambda);
        return result;
    }
}
