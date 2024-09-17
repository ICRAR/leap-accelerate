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

#include "DeviceIntegration.h"
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>

namespace icrar
{
namespace cuda
{
    DeviceIntegration::DeviceIntegration(
        int integrationNumber,
        Eigen::DSizes<Eigen::DenseIndex, 3> uvwShape,
        Eigen::DSizes<Eigen::DenseIndex, 4> visShape)
    : m_integrationNumber(integrationNumber)
    , m_uvws(uvwShape)
    , m_visibilities(visShape)
    {
    }

    DeviceIntegration::DeviceIntegration(const icrar::cpu::Integration& integration)
    : m_integrationNumber(integration.GetIntegrationNumber())
    , m_uvws(integration.GetUVW())
    , m_visibilities(integration.GetVis())
    {
    }

    __host__ void DeviceIntegration::Set(const DeviceIntegration& integration)
    {
        if(m_uvws.GetSize() != integration.GetUVW().GetSize())
        {
            throw icrar::invalid_argument_exception("uvw", "integration", __FILE__, __LINE__);
        }
        m_uvws.SetDataAsync(integration.m_uvws);

        if(m_visibilities.GetSize() != integration.m_visibilities.GetSize())
        {
            std::ostringstream os;
            os << "tensor size mismatch: this " << m_visibilities.GetDimensions() << "(" << m_visibilities.GetSize() << ")" << "\n";
            os << "other " << integration.m_visibilities.GetDimensions() << "(" << integration.m_visibilities.GetSize() << ")";
            throw icrar::invalid_argument_exception(os.str(), "integration", __FILE__, __LINE__);
        }
        m_visibilities.SetDataAsync(integration.m_visibilities);
    }

    __host__ void DeviceIntegration::Set(const icrar::cpu::Integration& integration)
    {
        if(boost::numeric_cast<int64_t>(m_uvws.GetSize()) != integration.GetUVW().size())
        {
            throw icrar::invalid_argument_exception("uvw", "integration", __FILE__, __LINE__);
        }
        m_uvws.SetDataAsync(integration.GetUVW().data());

        if(boost::numeric_cast<int64_t>(m_visibilities.GetSize()) != integration.GetVis().size())
        {
            std::ostringstream os;
            os << "tensor size mismatch: device " << m_visibilities.GetDimensions() << "(" << m_visibilities.GetSize() << ")" << "\n";
            os << "cpu " << integration.GetVis().dimensions() << "(" << integration.GetVis().size() << ")";
            throw icrar::invalid_argument_exception(os.str(), "integration", __FILE__, __LINE__);
        }

        m_visibilities.SetDataAsync(integration.GetVis().data());
    }

    __host__ void DeviceIntegration::ToHost(cpu::Integration&) const
    {
        throw icrar::not_implemented_exception(__FILE__, __LINE__);
        //m_visibilities.ToHost(host.m_data); //TODO(calgray): unsupported constant variant!
    }
} // namespace cuda
} // namespace icrar

#endif // CUDA_ENABLED