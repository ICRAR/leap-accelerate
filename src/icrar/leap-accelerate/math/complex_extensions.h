
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

#include <complex>

namespace icrar
{
    /**
     * @brief returns true if the magnitude of the difference between two
     * values are approximately equal (within the specified threshold)
     * 
     * @tparam T 
     * @param lhs left value
     * @param rhs right value
     * @param threshold  
     * @return true if left value approximately equals right value
     * @return false if left value does not approximately equals right value
     */
    template<typename T>
    bool isApprox(const std::complex<T>& lhs, const std::complex<T>& rhs, T threshold)
    {
        return std::abs(lhs - rhs) < threshold;
    }
} // namespace icrar