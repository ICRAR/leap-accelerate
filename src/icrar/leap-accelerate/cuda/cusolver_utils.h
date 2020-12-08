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

// TODO(calgray): cusolver must be included before helper_cuda
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>

#include <icrar/leap-accelerate/exception/exception.h>

#include <string>

namespace icrar
{
namespace cuda
{
    inline void cusolveSafeCall(cusolverStatus_t err, const char* func, const char* file, const int line)
    {
        if(CUSOLVER_STATUS_SUCCESS != err)
        {
            _cudaGetErrorEnum(err);
            //cudaDeviceReset();
            assert(0);
            throw icrar::exception("CUSOLVE error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n", file, line);
        }
    }
} // namespace cuda
} // namespace icrar

//#define cusolveSafeCall(val) ::icrar::cuda::_cusolveSafeCall((val), #val, __FILE__, __LINE__) // NOLINT(cppcoreguidelines-macro-usage)
