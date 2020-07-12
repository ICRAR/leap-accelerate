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

#include "vector.h"
#include "vector.cuh"

// template<typename T, int N>
// void h_add(const std::array<T, N>& a, const std::array<T, N>& b, std::array<T, N>& c)
// {
//     h_add(a.data(), b.data(), c.data(), a.size());
// }

//template void h_add<int, 1>(const std::array<int, 1>& a, const std::array<int, 1>& b, std::array<int, 1>& c) { h_add(a, b, c); }
//template void h_add<int, 1000>(const std::array<int, 1000>& a, const std::array<int, 1000>& b, std::array<int, 1000>& c) { h_add(a, b, c); }

void h_add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c) { h_add(a, b, c); }
void h_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) { h_add(a, b, c); }
void h_add(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) { h_add(a, b, c); }

void h_add(const casacore::Array<double>& a, const casacore::Array<double>& b, casacore::Array<double>& c) { h_add(a, b, c); }
void h_add(const casacore::Array<float>& a, const casacore::Array<float>& b, casacore::Array<float>& c) { h_add(a, b, c); }
void h_add(const casacore::Array<int>& a, const casacore::Array<int>& b, casacore::Array<int>& c) { h_add(a, b, c); }
