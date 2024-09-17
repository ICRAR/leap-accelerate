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

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/config.h>

#include <icrar/leap-accelerate/common/SphericalDirection.h>
#include <icrar/leap-accelerate/model/cpu/calibration/Calibration.h>

#include <icrar/leap-accelerate/algorithm/ILeapCalibrator.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

//#define EIGEN_HAS_CXX11 1
//#define EIGEN_VECTORIZE_GPU 1
//#define EIGEN_CUDACC 1
#include <Eigen/Core>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <boost/noncopyable.hpp>
#include <vector>


namespace icrar
{
    class MeasurementSet;
    namespace cpu
    {
        class Integration;
        class IntegrationResult;
        class BeamCalibration;
    }
    namespace cuda
    {
        class DeviceLeapData;
        class HostLeapData;
        class DeviceIntegration;
    }
}

namespace icrar
{
namespace cuda
{
    /**
     * @brief LEAP calibration object implemented using CUDA
     * 
     */
    class CudaLeapCalibrator : public ILeapCalibrator
    {
        cublasHandle_t m_cublasContext;
        cusolverDnHandle_t m_cusolverDnContext;

    public:
        CudaLeapCalibrator();
        ~CudaLeapCalibrator() override;
        
        /**
         * @copydoc ILeapCalibrator
         * Calibrates by performing phase rotation for each direction in @p directions
         * by splitting uvws and visibilities into integration batches per timestep.
         */
        void Calibrate(
            std::function<void(const cpu::Calibration&)> outputCallback,
            const icrar::MeasurementSet& ms,
            const std::vector<SphericalDirection>& directions,
            const Slice& solutionInterval,
            double minimumBaselineThreshold,
            bool computeCal1,
            boost::optional<unsigned int> referenceAntenna,
            const ComputeOptionsDTO& computeOptions) override;

        /**
         * @brief Calculates Ad into deviceAd, writes to cache if @p isFileSystemCacheEnabled is true
         * 
         * @param hostA matrix to invert
         * @param deviceA output device memory of A
         * @param hostAd output host memory of Ad (optionally written to)
         * @param deviceAd output device memory of Ad
         * @param isFileSystemCacheEnabled whether to use file caching
         * @param useCuda whether to use cuda solvers
         */
        void CalculateAd(
            HostLeapData& leapData,
            device_matrix<double>& deviceA,
            device_matrix<double>& deviceAd,
            bool isFileSystemCacheEnabled,
            bool useCuda);

        /**
         * @brief Calculates Ad1 into deviceAd1
         * 
         * @param hostA1 matrix to invert
         * @param deviceA1 output device memory of A1
         * @param hostAd1 output host memory of Ad1 (optionally written to)
         * @param deviceAd1 output device memory of Ad1
         */
        void CalculateAd1(
            HostLeapData& leapData,
            device_matrix<double>& deviceA1,
            device_matrix<double>& deviceAd1);

        /**
         * Performs only visibilities rotation on the GPU
         */
        void BeamCalibrate(
            const HostLeapData& hostMetadata,
            DeviceLeapData& deviceLeapData,
            const SphericalDirection& direction,
            cuda::DeviceIntegration& input,
            bool computeCal1,
            std::vector<cpu::BeamCalibration>& output_calibrations);
    };
} // namespace cuda
} // namespace icrar
#endif // CUDA_ENABLED
