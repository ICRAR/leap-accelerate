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

#include "matrix.h"
#include "matrix.cuh"

namespace icrar
{
namespace cuda
{
    void h_matrix(const casacore::Matrix<double>& a, const casacore::Matrix<double>& b, casacore::Matrix<double>& c) { h_multiply(a, b, c); }
    void h_matrix(const casacore::Matrix<float>& a, const casacore::Matrix<float>& b, casacore::Matrix<float>& c) { h_multiply(a, b, c); }
    void h_matrix(const casacore::Matrix<int>& a, const casacore::Matrix<int>& b, casacore::Matrix<int>& c) { h_multiply(a, b, c); }

    void h_matrix(const casacore::Matrix<double>& a, const casacore::Array<double>& b, casacore::Array<double>& c) { h_multiply(a, b, c); }
    void h_matrix(const casacore::Matrix<float>& a, const casacore::Array<float>& b, casacore::Array<float>& c) { h_multiply(a, b, c); }
    void h_matrix(const casacore::Matrix<int>& a, const casacore::Array<int>& b, casacore::Array<int>& c) { h_multiply(a, b, c); }
}
}