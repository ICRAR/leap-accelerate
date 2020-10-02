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

#include "DeviceIntegration.h"
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/model/cpu/Integration.h>

namespace icrar
{
namespace cuda
{
    DeviceIntegration::DeviceIntegration(Eigen::DSizes<Eigen::DenseIndex, 3> shape)
    : data(shape[0], shape[1], shape[2])
    , index(0)
    , x(0)
    , channels(0)
    , baselines(0)
    {

    }

    DeviceIntegration::DeviceIntegration(const icrar::cpu::Integration& integration)
    : data(integration.GetData())
    , index(integration.index)
    , x(integration.x)
    , channels(integration.channels)
    , baselines(integration.baselines)
    {

    }

    void DeviceIntegration::SetData(icrar::cpu::Integration& integration)
    {
        if(data.GetSize() != integration.GetData().size())
        {
            std::ostringstream os;
            os << "tensor size mismatch: device " << data.GetDimensions() << "(" << data.GetSize() << ")" << "\n";
            os << "cpu " << integration.GetData().dimensions() << "(" << integration.GetData().size() << ")";
            throw icrar::invalid_argument_exception(os.str(), "integration", __FILE__, __LINE__);
        }

        data.SetDataAsync(integration.GetData().data());
        index = integration.index;
        x = integration.x;
        channels = integration.channels;
        baselines = integration.baselines;
    }

    // void DeviceIntegration::ToHost(cpu::Integration& host) const
    // {
    //     data.ToHost(host.GetData());
    //     host.index = index;
    //     host.x = x;
    //     host.channels = channels;
    //     host.baselines = baselines;
    // }
}
}