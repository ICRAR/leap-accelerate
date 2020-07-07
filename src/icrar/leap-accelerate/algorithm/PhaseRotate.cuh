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

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <eigen3/Eigen/Core>

template<typename T>
__device__ void d_PhaseMatrixFunction()
{
    
}

template<typename T>
__global__ void g_PhaseMatrixFunction(const T* x1, const T* x2, T* y, int refAnt)
{

}

template<typename T, int Rows, int Cols>
__global__ void g_MatrixMultiply(
    const Eigen::Matrix<T, Rows, Cols>& v1,
    const Eigen::Matrix<T, Rows, Cols>& v2,
    Eigen::Matrix<T, Rows, Cols>& result)
{
    result = v1 * v2;
}

template<typename T, int Rows, int Cols>
__host__ void h_MatrixMultiply(
    const Eigen::Matrix<T, Rows, Cols>& v1,
    const Eigen::Matrix<T, Rows, Cols>& v2,
    Eigen::Matrix<T, Rows, Cols>& result)
{
    result = v1 * v2;
    //g_MatrixMultiply<<<1,1>>>(v1, v2, result);
}