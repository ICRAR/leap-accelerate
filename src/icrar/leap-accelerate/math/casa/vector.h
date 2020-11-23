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

#include <icrar/leap-accelerate/math/cpu/vector.h>
#include <casacore/casa/Arrays.h>
#include <casacore/casa/Arrays/Array.h>

namespace icrar
{
namespace casalib
{
    template<typename T>
    void add(const casacore::Vector<T>& a, const casacore::Vector<T>& b, casacore::Vector<T>& c)
    {
        if (a.shape() != b.shape() && a.shape() != c.shape())
        {
            throw std::runtime_error("argument shapes must be equal");
        }

        add(a.size(), a.data(), b.data(), c.data());
    }

    template<typename T>
    void multiply(T a, const casacore::Vector<T>& b, casacore::Vector<T>& c)
    {
        for(int col = 0; col < b.shape()[0]; ++col)
        {
            c(col) = a * b(col);
        }
    }

    template<typename T>
    casacore::Vector<T> multiply(T a, const casacore::Vector<T>& b)
    {
        casacore::Vector<T> c(b.shape());
        multiply(a, b, c);
        return c;
    }

    template<typename T>
    casacore::Vector<T> arg(const casacore::Vector<std::complex<T>>& a)
    {
        casacore::Vector<T> c(a.shape());
        for(int col = 0; col < a.shape()[0]; ++col)
        {
            c(col) = std::arg(a(col));
        }
        return c;
    }
}
}
