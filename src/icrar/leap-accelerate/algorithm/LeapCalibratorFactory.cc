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

#include "LeapCalibratorFactory.h"
#include <icrar/leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cuda/CudaLeapCalibrator.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    std::unique_ptr<ILeapCalibrator> LeapCalibratorFactory::Create(ComputeImplementation impl)
    {
        if(impl == ComputeImplementation::cpu)
        {
            return std::make_unique<cpu::CpuLeapCalibrator>();
        }
        else if(impl == ComputeImplementation::cuda)
        {
#if CUDA_ENABLED
            return std::make_unique<cuda::CudaLeapCalibrator>();
#else
            throw invalid_argument_exception("CUDA_ENABLED build option not enabled", "impl", __FILE__, __LINE__);
#endif
        }
        else
        {
            throw invalid_argument_exception("invalid argument", "impl", __FILE__, __LINE__);
        }
    }
} // namespace icrar