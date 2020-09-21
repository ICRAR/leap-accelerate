
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
     * @tparam T The input vector type
     * @tparam R the output vector type
     * @param vector 
     * @return std::vector<R> 
     */
    template<typename R, typename T>
    std::vector<R> vector_map(const std::vector<T>& vector, std::function<R(const T&)> lambda)
    {
        // see https://stackoverflow.com/questions/33379145/equivalent-of-python-map-function-using-lambda
        auto result = std::vector<R>(vector.size()); //TODO: this populates with 0, O(n), need to reserve and use back_inserter
        std::transform(vector.cbegin(), vector.cend(), result.begin(), lambda);
        return result;
    }
}
