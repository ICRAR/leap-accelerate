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

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace icrar
{
    template<typename T>
    using Tensor3X = Eigen::Tensor<T, 3>;

    template<typename T>
    bool isApprox(const Tensor3X<T>& lhs, const Tensor3X<T>& rhs, double tolerance)
    {
        for(int col = 0; col < rhs.dimension(0); ++col)
        {
            for(int row = 0; row < rhs.dimension(1); ++row)
            {
                for(int depth = 0; depth < rhs.dimension(2); ++depth)
                {
                    if(abs(lhs(row, col, depth) - rhs(row, col, depth)) > tolerance)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }
}