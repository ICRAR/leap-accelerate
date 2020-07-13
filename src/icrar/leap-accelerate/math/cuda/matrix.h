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

#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Matrix.h>

// C++ Style interface (templates not supported)
namespace icrar
{
namespace cuda
{
    void multiply(const int m, const int n, double* mat, double* vec, double* out);
    void multiply(const int m, const int n, float* mat, float* vec, float* out);
    void multiply(const int m, const int n, int* mat, int* vec, int* out);

    casacore::Array<double> multiply(const casacore::Matrix<double>& a, const casacore::Array<double>& b);
    casacore::Array<float> multiply(const casacore::Matrix<float>& a, const casacore::Array<float>& b);
    casacore::Array<int> multiply(const casacore::Matrix<int>& a, const casacore::Array<int>& b);

    void multiply(const casacore::Matrix<double>& a, const casacore::Array<double>& b, casacore::Array<double>& c);
    void multiply(const casacore::Matrix<float>& a, const casacore::Array<float>& b, casacore::Array<float>& c);
    void multiply(const casacore::Matrix<int>& a, const casacore::Array<int>& b, casacore::Array<int>& c);
}
}
