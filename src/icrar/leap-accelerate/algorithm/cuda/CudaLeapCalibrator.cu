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

#include "CudaLeapCalibrator.h"
#include <icrar/leap-accelerate/algorithm/cuda/PhaseRotate.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/cuda/helper_cuda.cuh>
#include <icrar/leap-accelerate/cuda/cusolver_utils.h>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublasLt.h>

namespace icrar
{
namespace cuda
{
    CudaLeapCalibrator::CudaLeapCalibrator()
    {
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        if(deviceCount < 1)
        {
            throw icrar::exception("CUDA error: no devices supporting CUDA.", __FILE__, __LINE__);
        }

        checkCudaErrors(cublasLtCreate(&m_cublasLtCtx));
        checkCudaErrors(cusolverDnCreate(&m_cusolverDnCtx));
    }

    CudaLeapCalibrator::~CudaLeapCalibrator()
    {
        checkCudaErrors(cusolverDnDestroy(m_cusolverDnCtx));
        checkCudaErrors(cublasLtDestroy(m_cublasLtCtx));
        checkCudaErrors(cudaDeviceReset());
    }

    cpu::CalibrateResult CudaLeapCalibrator::Calibrate(
        const icrar::MeasurementSet& ms,
        const std::vector<MVDirection>& directions,
        double minimumBaselineThreshold,
        bool isFileSystemCacheEnabled)
    {
        return cuda::Calibrate(ms, directions, minimumBaselineThreshold, isFileSystemCacheEnabled, m_cusolverDnCtx);
    }
} // namespace cuda
} // namespace icrar
