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
    DeviceIntegration::DeviceIntegration(const icrar::cpu::Integration& integration)
    : data(integration.data)
    , index(integration.index)
    , x(integration.x)
    , channels(integration.channels)
    , baselines(integration.baselines)
    {

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