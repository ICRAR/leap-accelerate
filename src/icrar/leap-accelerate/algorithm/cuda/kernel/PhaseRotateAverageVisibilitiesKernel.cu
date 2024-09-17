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

#include "PhaseRotateAverageVisibilitiesKernel.h"
#include <icrar/leap-accelerate/math/cuda/math.cuh>
#include <icrar/leap-accelerate/math/cpu/math.h>
#include <icrar/leap-accelerate/math/cpu/eigen_extensions.h>

// Note: older sm architectures do not support atomic add

namespace icrar
{
namespace cuda
{
    /**
     * @brief Rotates visibilities in parallel for baselines and channels
     * @note Atomic operator required for writing to @p pAvgData
     * 
     * @param constants measurement set constants
     * @param dd direction dependent rotation 
     * @param UVW unrotated uvws
     * @param integrationData inout integration data 
     * @param rotAvgVis output rotAvgVis to increment
     */
    __global__ void g_PhaseRotateAverageVisibilities(
        const icrar::cpu::Constants constants,
        const Eigen::Matrix3d dd,
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> UVWs,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 4>> integrationData,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>> rotAvgVis);

    __host__ void PhaseRotateAverageVisibilities(DeviceIntegration& integration, DeviceLeapData& leapData)
    {
        const auto& constants = leapData.GetConstants(); 
        assert(constants.num_pols == integration.GetNumPolarizations());
        assert(constants.channels == integration.GetNumChannels());
        assert(constants.nbaselines == integration.GetNumBaselines());

        auto integrationDataMap = Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 4>>(
            reinterpret_cast<cuDoubleComplex*>(integration.GetVis().Get()),
            static_cast<int>(integration.GetVis().GetDimensionSize(0)), // inferring (const int) causes error
            static_cast<int>(integration.GetVis().GetDimensionSize(1)), // inferring (const int) causes error
            static_cast<int>(integration.GetVis().GetDimensionSize(2)), // inferring (const int) causes error
            static_cast<int>(integration.GetVis().GetDimensionSize(3)) // inferring (const int) causes error
        );

        auto rotAvgVisMap = Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>>(
            reinterpret_cast<cuDoubleComplex*>(leapData.GetAvgData().Get()),
            static_cast<int>(leapData.GetAvgData().GetRows()), // inferring (const int) causes error
            static_cast<int>(leapData.GetAvgData().GetCols()) // inferring (const int) causes error
        );

        const auto UVWMap = Eigen::TensorMap<const Eigen::Tensor<double, 3>>(
            integration.GetUVW().Get(),
            integration.GetUVW().GetDimensionSize(0),
            integration.GetUVW().GetDimensionSize(1),
            integration.GetUVW().GetDimensionSize(2)
        );

        dim3 blockSize = dim3(8, 128, 1); // block size can be any value where the product is <=1024
        dim3 gridSize = dim3(
            cpu::ceil_div<int64_t>(integration.GetNumChannels(), blockSize.x),
            cpu::ceil_div<int64_t>(integration.GetNumBaselines(), blockSize.y),
            cpu::ceil_div<int64_t>(integration.GetNumTimesteps(), blockSize.z)
        );
        g_PhaseRotateAverageVisibilities<<<gridSize, blockSize>>>(
            constants,
            leapData.GetDD(),
            UVWMap,
            integrationDataMap,
            rotAvgVisMap);
        checkCudaErrors(cudaGetLastError());
    }

    __global__ void g_PhaseRotateAverageVisibilities(
        const icrar::cpu::Constants constants,
        const Eigen::Matrix3d dd,
        const Eigen::TensorMap<const Eigen::Tensor<double, 3>> UVWs,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 4>> integrationData,
        Eigen::TensorMap<Eigen::Tensor<cuDoubleComplex, 2>> rotAvgVis)
    {
        const int integration_polarizations = integrationData.dimension(0);
        const int integration_channels = integrationData.dimension(1);
        const int integration_baselines = integrationData.dimension(2);
        const int integration_timesteps = integrationData.dimension(3);

        constexpr double two_pi = 2 * CUDART_PI;

        //parallel execution per channel
        int channel = blockDim.x * blockIdx.x + threadIdx.x;
        int baseline = blockDim.y * blockIdx.y + threadIdx.y;
        int timestep = blockDim.z * blockIdx.z + threadIdx.z;

        if(baseline < integration_baselines && channel < integration_channels)
        {
            // Rotation
            auto uvw = Eigen::Vector3d(UVWs(0, baseline, timestep), UVWs(1, baseline, timestep), UVWs(2, baseline, timestep));
            Eigen::Vector3d rotatedUVW = dd * uvw;
            double shiftFactor = -two_pi * (rotatedUVW.z() - uvw.z());
            double shiftRad = shiftFactor / constants.GetChannelWavelength(channel);
            cuDoubleComplex shiftCoeff = cuCexp(make_cuDoubleComplex(0.0, shiftRad));
            for(int polarization = 0; polarization < integration_polarizations; polarization++)
            {
                integrationData(polarization, channel, baseline, timestep)
                = cuCmul(integrationData(polarization, channel, baseline, timestep), shiftCoeff);
            }

            // Averaging
            bool hasNaN = false;
            for(int polarization = 0; polarization < integration_polarizations; polarization++)
            {
                cuDoubleComplex n = integrationData(polarization, channel, baseline, timestep);
                hasNaN |= isnan(n.x) || isnan(n.y);
            }
            if(!hasNaN)
            {
                // XX + YY
                // .x -> real
                // .y -> imag
                atomicAdd(&rotAvgVis(baseline, 0).x, integrationData(0, channel, baseline, timestep).x);
                atomicAdd(&rotAvgVis(baseline, 0).y, integrationData(0, channel, baseline, timestep).y);
                atomicAdd(&rotAvgVis(baseline, 0).x, integrationData(integration_polarizations - 1, channel, baseline, timestep).x);
                atomicAdd(&rotAvgVis(baseline, 0).y, integrationData(integration_polarizations - 1, channel, baseline, timestep).y);
            }
        }
    }
} // namespace cuda
} // namespace icrar
