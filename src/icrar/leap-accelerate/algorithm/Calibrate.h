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

#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/algorithm/gpu/PhaseRotate.h>

#include <icrar/leap-accelerate/core/compute_implementation.h>
#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
    class MeasurementSet;

    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class CalibrationResult;
    }

    Calibrate(
        ComputeImplementation impl,
        const icrar::MeasurementSet& ms,
        const std::vector<icrar::MVDirection>& directions,
        double minimumBaselineThreshold,
        bool isFileSystemCacheEnabled)
    {
        if(impl == ComputeImplementation::cpu)
        {
            return cpu::Calibrate(ms, directions, minimumBaselineThreshold, isFileSystemCacheEnabled);
        }
        else if(impl == ComputeImplementation::cuda)
        {
#ifdef CUDA_ENABLED
            return cuda::Calibrate(ms, directions, minimumBaselineThreshold, isFileSystemCacheEnabled);
#else
            throw invalid_argument_exception("cuda build option not enabled", "impl", __FILE__, __LINE__);
#endif
        }
        else
        {
            throw invalid_argument_exception("invalid argument", "impl", __FILE__, __LINE__);
        }
    }
}