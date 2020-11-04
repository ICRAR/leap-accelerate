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

#include "Integration.h"
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/ms/utils.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>

#include <icrar/leap-accelerate/core/logging.h>

namespace icrar
{
namespace cpu
{
    Integration::Integration(const icrar::casalib::Integration& integration)
    : m_integrationNumber(integration.integration_number)
    , index(integration.index)
    , x(integration.x)
    , channels(integration.channels)
    , baselines(integration.baselines)
    , m_uvw(ToUVWVector(integration.uvw))
    {
        m_data = Eigen::Tensor<std::complex<double>, 3>(integration.data);
    }

    Integration::Integration(unsigned int integrationNumber, const icrar::MeasurementSet& ms)
    : Integration(integrationNumber, ms, 0, ms.GetNumChannels(), ms.GetNumBaselines(), ms.GetNumPols())
    { }

    Integration::Integration(
        unsigned int integrationNumber,
        const icrar::MeasurementSet& ms,
        unsigned int startBaseline,
        unsigned int channels,
        unsigned int baselines,
        unsigned int polarizations)
    : m_integrationNumber(integrationNumber)
    , index(startBaseline)
    , x(baselines)
    , channels(channels)
    , baselines(baselines)
    {
        constexpr int startChannel = 0;
        size_t vis_size = (baselines - startBaseline) * (channels - startChannel) * polarizations * sizeof(std::complex<double>);
        BOOST_LOG_TRIVIAL(info) << "vis:" << vis_size / (1024.0 * 1024.0 * 1024.0) << " GiB";
        size_t uvw_size = (baselines - startBaseline) * 3;
        BOOST_LOG_TRIVIAL(info) << "uvw:" << uvw_size / (1024.0 * 1024.0 * 1024.0) << " GiB";
        m_data = ms.GetVis(startBaseline, startChannel, channels, baselines, polarizations);
        m_uvw = ToUVWVector(ms.GetCoords(startBaseline, baselines));
    }

    bool Integration::operator==(const Integration& rhs) const
    {
        Eigen::Map<const Eigen::VectorXcd> datav(m_data.data(), m_data.size());
        Eigen::Map<const Eigen::VectorXcd> rhsdatav(rhs.m_data.data(), rhs.m_data.size());
        
        return datav.isApprox(rhsdatav)
        && m_uvw == rhs.m_uvw
        && m_integrationNumber == rhs.m_integrationNumber;
    }

    const std::vector<icrar::MVuvw>& Integration::GetUVW() const
    {
        return m_uvw;
    }
}
}
