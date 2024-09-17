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

#include <icrar/leap-accelerate/model/cpu/Integration.h>
#include <cuda_runtime.h>

namespace icrar
{
namespace cuda
{
    /**
     * @brief A cuda decorator for cpu::Integration. 
     * This class stores data on the host using pinned memory to allow for asyncronous read and write with cuda.
     */
    class HostIntegration : public cpu::Integration
    {
    public:
        HostIntegration(
            int integrationNumber,
            Eigen::Tensor<double, 3>&& uvws,
            Eigen::Tensor<std::complex<double>, 4>&& visibilities)
        : Integration(
                integrationNumber,
                std::move(uvws),
                std::move(visibilities)
            )
        {
            cudaHostRegister(m_visibilities.data(), m_visibilities.size() * sizeof(decltype(*m_visibilities.data())), cudaHostRegisterPortable);
            cudaHostRegister(m_UVW.data(), m_UVW.size() * sizeof(decltype(*m_UVW.data())), cudaHostRegisterPortable);
        }

        static HostIntegration CreateFromMS(
            const icrar::MeasurementSet& ms,
            int integrationNumber,
            const Slice& timestepSlice,
            const Slice& polarizationSlice = Slice(0, boost::none, 1)
        )
        {
            // assert(timestepSlice.GetInterval() == 1);

            uint32_t channels = ms.GetNumChannels();
            uint32_t baselines = ms.GetNumBaselines();
            uint32_t polarizations = ms.GetNumPols();

            constexpr int startChannel = 0;
            size_t vis_size = baselines * (channels - startChannel) * (polarizations > 1 ? 2 : 1) * sizeof(std::complex<double>);
            LOG(info) << "vis: " << memory_amount(vis_size);
            size_t uvw_size = baselines * 3 * sizeof(double);
            LOG(info) << "uvw: " << memory_amount(uvw_size);

            return HostIntegration(
                integrationNumber,
                ms.ReadCoords(timestepSlice),
                ms.ReadVis(timestepSlice, polarizationSlice)
            );
        }

        ~HostIntegration()
        {
            cudaHostUnregister(m_visibilities.data());
            cudaHostUnregister(m_UVW.data());
        }
    };
}
}
#endif // CUDA_ENABLED
