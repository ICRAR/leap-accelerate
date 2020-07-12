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

#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <casacore/casa/Arrays/Matrix.h>

namespace icrar
{
    template<typename T>
    void h_multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b, casacore::Matrix<T>& c)
    {
        //TODO
    }

    template<typename T>
    casacore::Matrix<T> h_multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b)
    {
        casacore::Matrix<T> c;
        h_multiply(a, b, c);
        return c;
    }

    template<typename T>
    void h_multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b, casacore::Array<T>& c)
    {
        //TODO
    }

    template<typename T>
    casacore::Array<T> h_multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b)
    {
        auto c = casacore::Array<T>();
        h_multiply(a, b, c);
        return c;
    }
}