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

#include <icrar/leap-accelerate/math/vector.cuh>

__global__ void addi(const int* x1, const int* x2, int* y)
{
    d_add(x1, x2, y);
}


//__global__ void addf(const float* x1, const float* x2, float* y)
//{
//    add(x1, x2, y);
//}

//extern "C"
//{
//    __global__ void addi(const int* x1, const int* x2, int* y);
//
//    __global__ void addf(const float* x1, const float* x2, float* y);
//
//    __global__ void addd(const double* x1, const double* x2, double* y);
//}