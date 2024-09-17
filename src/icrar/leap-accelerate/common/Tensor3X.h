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

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace icrar
{
    template<typename T>
    using Tensor3X = Eigen::Tensor<T, 3>;

    template<typename T>
    bool isApprox(const Tensor3X<T>& lhs, const Tensor3X<T>& rhs, double tolerance)
    {
        auto dimsEqual = [&]() { return lhs.dimensions() == rhs.dimensions(); };
        auto dataEqual = [&]()
        {
            // TODO(calgray): optimize, try
            // return std::inner_product(lhs.data(), lhs.data() + lhs.size(), rhs.data(), true, std::logical_and<bool>, [&tolerance](auto e1, auto e2) {
            //     return std::abs(e1 - e2) > tolerance;
            // });

            for(int x = 0; x < rhs.dimension(0); ++x)
            {
                for(int y = 0; y < rhs.dimension(1); ++y)
                {
                    for(int z = 0; z < rhs.dimension(2); ++z)
                    {
                        if(abs(lhs(x, y, z) - rhs(x, y, z)) > tolerance)
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        return dimsEqual() && dataEqual();
    }
} // namespace icrar