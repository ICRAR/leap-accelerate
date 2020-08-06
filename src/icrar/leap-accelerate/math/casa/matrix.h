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

#include <icrar/leap-accelerate/math/linear_math_helper.h>
#include <casacore/casa/Arrays/Matrix.h>

#ifdef __CUDACC_VER__
#undef __CUDACC_VER__
#define __CUDACC_VER__ ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif
#include <eigen3/Eigen/Core>

namespace icrar
{
namespace casalib
{
    template<typename T>
    void multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b, casacore::Matrix<T>& c)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ea = ConvertMatrix(a);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eb = ConvertMatrix(b);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ec = ea * eb;
        c = ConvertMatrix(ec);
    }

    template<typename T>
    casacore::Matrix<T> multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b)
    {
        casacore::Matrix<T> c;
        multiply(a, b, c);
        return c;
    }

    template<typename T>
    void multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b, casacore::Array<T>& c)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ea = ConvertMatrix(a);
        Eigen::Matrix<T, Eigen::Dynamic, 1> eb = ConvertVector(b);
        Eigen::Matrix<T, Eigen::Dynamic, 1> ec = ea * eb;
        c = ConvertVector(ec);
    }

    template<typename T>
    casacore::Array<T> multiply(const casacore::Matrix<T>& a, const casacore::Array<T>& b)
    {
        auto c = casacore::Array<T>();
        multiply(a, b, c);
        return c;
    }
}
}