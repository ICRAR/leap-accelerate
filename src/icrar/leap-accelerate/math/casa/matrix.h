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

#include <icrar/leap-accelerate/math/math_conversion.h>
#include <casacore/casa/Arrays/Matrix.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <Eigen/Core>

namespace icrar
{
namespace casalib
{
    template<typename T>
    void multiply(const casacore::Matrix<T>& a, const casacore::Matrix<T>& b, casacore::Matrix<T>& c)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ea = ToMatrix(a);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eb = ToMatrix(b);
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
    void multiply(const casacore::Matrix<T>& a, const casacore::Vector<T>& b, casacore::Vector<T>& c)
    {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ea = ToMatrix(a);
        Eigen::Matrix<T, Eigen::Dynamic, 1> eb = ToVector(b);
        Eigen::Matrix<T, Eigen::Dynamic, 1> ec = ea * eb;
        c = ConvertVector(ec);
    }

    template<typename T>
    casacore::Vector<T> multiply(const casacore::Matrix<T>& a, const casacore::Vector<T>& b)
    {
        casacore::Vector<T> c;
        multiply(a, b, c);
        return c;
    }
}
}