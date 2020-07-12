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

#include <casacore/casa/Arrays/Array.h>

#include <array>
#include <vector>
#include <stdexcept>

namespace icrar
{
namespace cpu
{
    /**
    * @brief Performs vector addition of equal length vectors
    *
    * @tparam T vector value type
    * @param x1 left vector
    * @param x2 right vector
    * @param y out vector
    */
    template<typename T>
    void add(size_t n, const T* x1, const T* x2, T* y)
    {
        for(size_t i; i < n; i++)
        {
            y[i] = x1[i] + x2[i];
        }
    }

    template<typename T, size_t N>
    void add(const std::array<T, N>& a, const std::array<T, N>& b, std::array<T, N>& c)
    {
        add(a.size(), a.data(), b.data(), c.data());
    }

    template<typename T, size_t N>
    std::array<T, N> add(const std::array<T, N>& a, const std::array<T, N>& b)
    {
        std::array<T, N> result;
        add(a.size(), a.data(), b.data(), result.data());
        return result;
    }

    template<typename T>
    void add(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c)
    {
        if (a.size() != b.size() && a.size() != c.size())
        {
            throw std::runtime_error("argument sizes must be equal");
        }

        add(a.size(), a.data(), b.data(), c.data());
    }

    template<typename T>
    void add(const casacore::Array<T>& a, const casacore::Array<T>& b, casacore::Array<T>& c)
    {
        if (a.shape() != b.shape() && a.shape() != c.shape())
        {
            throw std::runtime_error("argument shapes must be equal");
        }

        add(a.shape()[0], a.data(), b.data(), c.data());
    }
}
}