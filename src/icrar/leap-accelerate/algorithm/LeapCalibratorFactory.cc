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

#include "LeapCalibratorFactory.h"
#include <icrar/leap-accelerate/algorithm/cpu/CpuLeapCalibrator.h>
#include <icrar/leap-accelerate/algorithm/cuda/CudaLeapCalibrator.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    std::unique_ptr<ILeapCalibrator> LeapCalibratorFactory::Create(ComputeImplementation impl) const
    {
        if(impl == ComputeImplementation::cpu)
        {
            return std::make_unique<CpuLeapCalibrator>();
        }
        else if(impl == ComputeImplementation::cuda)
        {
#ifdef CUDA_ENABLED
            return std::make_unique<CudaLeapCalibrator>();
#else
            throw invalid_argument_exception("cuda build option not enabled", "impl", __FILE__, __LINE__);
#endif
        }
        else
        {
            throw invalid_argument_exception("invalid argument", "impl", __FILE__, __LINE__);
        }
    }
} // namespace icrar