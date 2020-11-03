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

#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/MVuvw.h>
#include <icrar/leap-accelerate/common/MVDirection.h>

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
        device_tensor3<std::complex<double>> m_visibilities; //[polarizations][baselines][channels]

        union
        {
            std::array<size_t, 4> m_parameters;
            struct
            {
                size_t index;
                size_t x;
                size_t channels;
                size_t baselines;
            };
        };
    public:
        /**
         * @brief Construct a new Device Integration object where visibilities is a zero tensor of @shape 
         * 
         * @param shape 
         */
        DeviceIntegration(int integrationNumber, Eigen::DSizes<Eigen::DenseIndex, 3> shape);

        /**
         * @brief Construct a new Device Integration object with a data syncronous copy
         * 
         * @param integration 
         */
        DeviceIntegration(const icrar::cpu::Integration& integration);

        int GetIntegrationNumber() const { return m_integrationNumber; }
        size_t GetIndex() const { return index; }
        //size_t GetX() const { return x; }
        size_t GetChannels() const { return channels; }
        size_t GetBaselines() const { return baselines; }
        
        const device_tensor3<std::complex<double>>& GetVis() const { return m_visibilities; }
        device_tensor3<std::complex<double>>& GetVis() { return m_visibilities; }

        /**
         * @brief Set the Data object
         * 
         * @param integration 
         */
        void SetData(const icrar::cpu::Integration& integration);

        /**
         * @brief Copies device data to a host object
         * 
         * @param host object with data on cpu memory
         */
        void ToHost(cpu::Integration& host) const;
    };
}
}
