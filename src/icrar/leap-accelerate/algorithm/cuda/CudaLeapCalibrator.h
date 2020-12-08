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
#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>
#include <boost/noncopyable.hpp>

#include <cusolverDn.h>
#include <cublasLt.h>

#include <vector>

namespace icrar
{
namespace cuda
{
    class CudaLeapCalibrator : public ILeapCalibrator
    {
        cublasLtHandle_t m_cublasLtCtx = nullptr;
        cusolverDnHandle_t m_cusolverDnCtx = nullptr; 
    public:
        CudaLeapCalibrator();
        ~CudaLeapCalibrator() override;

        virtual cpu::CalibrateResult Calibrate(
            const icrar::MeasurementSet& ms,
            const std::vector<MVDirection>& directions,
            double minimumBaselineThreshold,
            bool isFileSystemCacheEnabled) override;
    };
} // namespace cuda
} // namespace icrar
#endif
