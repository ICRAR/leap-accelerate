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
#include <icrar/leap-accelerate/math/linear_math_helper.h>
#include <icrar/leap-accelerate/ms/utils.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>

namespace icrar
{
namespace cpu
{
    Integration::Integration(const icrar::MeasurementSet& ms, int integrationNumber, int channels, int baselines, int polarizations, int uvws)
    : integration_number(integrationNumber)
    , index(0)
    , x(0)
    , channels(channels)
    , baselines(baselines)
    {
        data = ms.GetVis(channels, baselines, polarizations);
        uvw = ToCasaUVWVector(ms.GetCoords(index, baselines));
    }

    bool Integration::operator==(const Integration& rhs) const
    {
        Eigen::Map<const Eigen::VectorXcd> datav(data.data(), data.size());
        Eigen::Map<const Eigen::VectorXcd> rhsdatav(rhs.data.data(), rhs.data.size());
        
        return datav.isApprox(rhsdatav)
        && uvw == rhs.uvw
        && integration_number == rhs.integration_number;
    }
}
}
