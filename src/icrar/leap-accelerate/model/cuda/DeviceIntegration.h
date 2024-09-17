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

#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/SphericalDirection.h>

#include <icrar/leap-accelerate/model/cpu/MVuvw.h>
#include <icrar/leap-accelerate/common/constants.h>

#include <icrar/leap-accelerate/cuda/device_tensor.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <boost/optional.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

#include <cuComplex.h>

namespace icrar
{
namespace cpu
{
    class Integration;
}
}

namespace icrar
{
namespace cuda
{
    /**
     * @brief A Cuda memory buffer instance of visibility data for integration
     * 
     */
    class DeviceIntegration
    {
        int m_integrationNumber;

        device_tensor3<double> m_uvws; // [3][baselines][timesteps]
        device_tensor4<std::complex<double>> m_visibilities; //[polarizations][channels][baselines][timesteps]
        
    public:
        /**
         * @brief Construct a new Device Integration object where visibilities is a zero tensor of @shape 
         * 
         * @param shape 
         */
        DeviceIntegration(int integrationNumber, Eigen::DSizes<Eigen::DenseIndex, 3> uvwShape, Eigen::DSizes<Eigen::DenseIndex, 4> visShape);

        /**
         * @brief Construct a new Device Integration object with a data syncronous copy
         * 
         * @param integration 
         */
        DeviceIntegration(const icrar::cpu::Integration& integration);

        /**
         * @brief Set the Data object
         * 
         * @param integration 
         */
        __host__ void Set(const icrar::cpu::Integration& integration);

        /**
         * @brief Set the Data object
         * 
         * @param integration 
         */
        __host__ void Set(const icrar::cuda::DeviceIntegration& integration);

        int GetIntegrationNumber() const { return m_integrationNumber; }

        size_t GetNumPolarizations() const { return m_visibilities.GetDimensionSize(0); }
        size_t GetNumChannels() const { return m_visibilities.GetDimensionSize(1); }
        size_t GetNumBaselines() const { return m_visibilities.GetDimensionSize(2); }
        size_t GetNumTimesteps() const { return m_visibilities.GetDimensionSize(3); }

        const device_tensor3<double>& GetUVW() const { return m_uvws; }

        const device_tensor4<std::complex<double>>& GetVis() const { return m_visibilities; }
        device_tensor4<std::complex<double>>& GetVis() { return m_visibilities; }

        /**
         * @brief Copies device data to a host object
         * 
         * @param host object with data on cpu memory
         */
        __host__ void ToHost(cpu::Integration& host) const;
    };
} // namespace cuda
} // namepace icrar

#endif // CUDA_ENABLED
