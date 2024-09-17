
/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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

__device__ __forceinline__ double cuCarg(cuDoubleComplex z)
{
    return atan2(cuCreal(z), cuCimag(z));
}
