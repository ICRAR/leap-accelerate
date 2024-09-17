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

#include "Integration.h"
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>

#include <icrar/leap-accelerate/core/memory/ioutils.h>
#include <icrar/leap-accelerate/core/log/logging.h>

namespace icrar
{
namespace cpu
{
    Integration::Integration(
        int integrationNumber,
        Eigen::Tensor<double, 3>&& uvws,
        Eigen::Tensor<std::complex<double>, 4>&& visibilities)
    : m_integrationNumber(integrationNumber)
    , m_UVW(std::move(uvws))
    , m_visibilities(std::move(visibilities))
    {
    }

    Integration Integration::CreateFromMS(
        const icrar::MeasurementSet& ms,
        int integrationNumber,
        const Slice& timestepSlice,
        const Slice& polarizationSlice
    )
    {
        //assert(timestepSlice.GetInterval() == 1);

        uint32_t channels = ms.GetNumChannels();
        uint32_t baselines = ms.GetNumBaselines();
        uint32_t polarizations = ms.GetNumPols();

        constexpr int startChannel = 0;
        size_t vis_size = baselines * (channels - startChannel) * (polarizations > 1 ? 2 : 1) * sizeof(std::complex<double>);
        LOG(info) << "vis: " << memory_amount(vis_size);
        size_t uvw_size = baselines * 3 * sizeof(double);
        LOG(info) << "uvw: " << memory_amount(uvw_size);

        return Integration(
            integrationNumber,
            ms.ReadCoords(timestepSlice),
            ms.ReadVis(timestepSlice, polarizationSlice)
        );
    }

    bool Integration::operator==(const Integration& rhs) const
    {
        Eigen::Map<const Eigen::ArrayXcd> datav(m_visibilities.data(), m_visibilities.size());
        Eigen::Map<const Eigen::ArrayXcd> rhsdatav(rhs.m_visibilities.data(), rhs.m_visibilities.size());
        
        return 
            m_integrationNumber == rhs.m_integrationNumber
            && m_visibilities.dimensions() == rhs.m_visibilities.dimensions()
            && m_UVW.dimensions() == rhs.m_UVW.dimensions()
            && datav.isApprox(rhsdatav); //TODO(calgray) compare UVWs
    }
} // namespace cpu
} // namespace icrar
