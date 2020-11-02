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

#include "DeviceMetaData.h"
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cuda
{
    DeviceMetaData::DeviceMetaData(const cpu::MetaData& metadata)
    : constants(metadata.GetConstants())
    , UVW(metadata.m_UVW)
    , oldUVW(metadata.m_oldUVW)
    , dd(metadata.dd)
    , direction(metadata.direction)
    , avg_data(metadata.avg_data)
    , A(metadata.GetA())
    , I(metadata.GetI())
    , Ad(metadata.GetAd())
    , A1(metadata.GetA1())
    , I1(metadata.GetI1())
    , Ad1(metadata.GetAd1())
    {
    }

    const icrar::cpu::Constants& DeviceMetaData::GetConstants()
    {
        return constants;
    }

    void DeviceMetaData::ToHost(cpu::MetaData& metadata) const
    {
        metadata.m_constants = constants;

        A.ToHost(metadata.m_A);
        I.ToHost(metadata.m_I);
        Ad.ToHost(metadata.m_Ad);
        A1.ToHost(metadata.m_A1);
        I1.ToHost(metadata.m_I1);
        Ad1.ToHost(metadata.m_Ad1);

        oldUVW.ToHost(metadata.m_oldUVW);
        UVW.ToHost(metadata.m_UVW);
        metadata.direction = direction;
        metadata.dd = dd;
        avg_data.ToHost(metadata.avg_data);
    }

    void DeviceMetaData::AvgDataToHost(Eigen::MatrixXcd& host) const
    {
        avg_data.ToHost(host);
    }

    cpu::MetaData DeviceMetaData::ToHost() const
    {
        //TODO: tidy up using a constructor for now
        //TODO: casacore::MVuvw and casacore::MVDirection not safe to copy to cuda
        std::vector<icrar::MVuvw> uvwTemp;
        UVW.ToHost(uvwTemp);
        cpu::MetaData result = cpu::MetaData();
        ToHost(result);
        return result;
    }

    void DeviceMetaData::ToHostAsync(cpu::MetaData& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}