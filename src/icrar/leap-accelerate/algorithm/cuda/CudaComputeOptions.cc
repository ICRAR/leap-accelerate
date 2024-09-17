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

#if CUDA_ENABLED

#include <icrar/leap-accelerate/algorithm/cuda/CudaComputeOptions.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/core/log/logging.h>
#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/common/Range.h>

#include <icrar/leap-accelerate/cuda/cuda_info.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/core/memory/system_memory.h>
#include <boost/numeric/conversion/cast.hpp>

#include <boost/optional.hpp>

namespace icrar
{
    CudaComputeOptions::CudaComputeOptions(const ComputeOptionsDTO& computeOptions, const icrar::MeasurementSet& ms, const Rangei& solutionRange)
    {
        LOG(info) << "Determining cuda compute options";

        size_t free = GetAvailableCudaPhysicalMemory();
        size_t VisSize = solutionRange.GetInterval() * ms.GetNumPols() * ms.GetNumBaselines() * ms.GetNumChannels() * sizeof(std::complex<double>);
        size_t ASize = ms.GetNumStations() * ms.GetNumBaselines() * sizeof(double);
        double safetyFactor = 1.3;

        if(computeOptions.isFileSystemCacheEnabled.is_initialized())
        {
            isFileSystemCacheEnabled = computeOptions.isFileSystemCacheEnabled.get();
        }
        else
        {
            isFileSystemCacheEnabled = false;
        }

        if(computeOptions.useCusolver.is_initialized())
        {
            useCusolver = computeOptions.useCusolver.get();
        }
        else // determine from available memory
        {
            // A, Ad and SVD buffers required to compute inverse
            auto required = boost::numeric_cast<size_t>(static_cast<double>(3 * ASize) * safetyFactor);
            if(required < free)
            {
                LOG(info) << memory_amount(free) << " > " << memory_amount(required) << ". Enabling Cusolver";
                useCusolver = true;
            }
            else
            {
                LOG(info) << memory_amount(free) << " < " << memory_amount(required) << ". Disabling Cusolver";
                useCusolver = false;
            }
        }

        if(computeOptions.useIntermediateBuffer.is_initialized())
        {
            useIntermediateBuffer = computeOptions.useIntermediateBuffer.get();
        }
        else // determine from available memory
        {
            // A, Ad and 2x visibilities required to calibrate
            auto required = boost::numeric_cast<size_t>(static_cast<double>(2 * ASize + 2 * VisSize) * safetyFactor);
            if(required < free)
            {
                LOG(info) << memory_amount(free) << " > " << memory_amount(required) << ". Enabling IntermediateBuffer";
                useIntermediateBuffer = true;
            }
            else
            {
                LOG(info) << memory_amount(free) << " < " << memory_amount(required) << ". Disabling IntermediateBuffer";
                useIntermediateBuffer = false;
            }
        }
    }
} // namespace icrar

#endif // CUDA_ENABLED