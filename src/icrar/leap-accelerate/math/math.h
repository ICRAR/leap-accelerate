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

#include <icrar/leap-accelerate/common/MVDirection.h>
#include <vector>

namespace icrar
{
    /**
     * @brief returns a linear sequence of values from @c start to @c stop at incrmements of @c step
     * 
     * @tparam IntType 
     * @param start 
     * @param stop 
     * @param step 
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop, IntType step)
    {
        if (step == IntType(0))
        {
            throw std::invalid_argument("step for range must be non-zero");
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
     * @brief 
     * 
     * @tparam IntType 
     * @param start 
     * @param stop 
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType start, IntType stop)
    {
        return range(start, stop, IntType(1));
    }

    /**
     * @brief 
     * 
     * @tparam IntType 
     * @param stop 
     * @return std::vector<IntType> 
     */
    template <typename IntType>
    std::vector<IntType> range(IntType stop)
    {
        return range(IntType(0), stop, IntType(1));
    }

    /**
     * @brief Converts a unit catersian direction to polar coordinates
     * 
     * @param cartesian 3-d cartesion coordinate/vector
     * @return Eigen::Vector2d 
     */
    Eigen::Vector2d ToPolar(const MVDirection& cartesian);
}