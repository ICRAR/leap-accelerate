
/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#pragma once

/**
 * @brief Computes the complex exponent of a complex value
 * 
 * @param z complex value
 * @return e ^ ( @p z )
 */
__device__ __forceinline__ cuDoubleComplex cuCexp(cuDoubleComplex z)
{
    // see https://forums.decuCexpveloper.nvidia.com/t/complex-number-exponential-function/24696/2
    double resx = 0.0;
    double resy = 0.0;
    double zx = cuCreal(z);
    double zy = cuCimag(z);

    sincos(zy, &resy, &resx);
    
    double t = exp(zx);
    resx *= t;
    resy *= t;
    return make_cuDoubleComplex(resx, resy);
}