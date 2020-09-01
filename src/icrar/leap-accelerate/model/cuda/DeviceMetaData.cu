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
    , UVW(metadata.UVW)
    , oldUVW(metadata.oldUVW)
    , dd(metadata.dd)
    , avg_data(metadata.avg_data)
    , A(metadata.GetA())
    , I(metadata.GetI())
    , Ad(metadata.GetAd())
    , A1(metadata.GetA1())
    , I1(metadata.GetI1())
    , Ad1(metadata.GetAd1())
    {

    }

    void DeviceMetaData::ToHost(cpu::MetaData& metadata) const
    {
        metadata.m_constants = constants;

        A.ToHost(metadata.A);
        I.ToHost(metadata.I);
        Ad.ToHost(metadata.Ad);
        A1.ToHost(metadata.A1);
        I1.ToHost(metadata.I1);
        Ad1.ToHost(metadata.Ad1);

        oldUVW.ToHost(metadata.oldUVW);
        UVW.ToHost(metadata.UVW);
        metadata.direction = direction;
        metadata.dd = dd;
        avg_data.ToHost(metadata.avg_data);
    }

    cpu::MetaData DeviceMetaData::ToHost() const
    {
        //TODO: tidy up using a constructor for now
        //TODO: casacore::MVuvw and casacore::MVDirection not safe to copy to cuda
        std::vector<icrar::MVuvw> uvwTemp;
        UVW.ToHost(uvwTemp);
        cpu::MetaData result = cpu::MetaData(casalib::MetaData(), casacore::MVDirection(), ToCasaUVWVector(uvwTemp));
        ToHost(result);
        return result;
    }

    void DeviceMetaData::ToHostAsync(cpu::MetaData& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}