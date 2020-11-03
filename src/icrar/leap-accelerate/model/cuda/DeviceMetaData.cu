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
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/exception/exception.h>

namespace icrar
{
namespace cuda
{
    ConstantMetaData::ConstantMetaData(
            const icrar::cpu::Constants constants,
            Eigen::MatrixXd A,
            Eigen::VectorXi I,
            Eigen::MatrixXd Ad,
            Eigen::MatrixXd A1,
            Eigen::VectorXi I1,
            Eigen::MatrixXd Ad1)
        : m_constants(constants)
        , m_A(A)
        , m_I(I)
        , m_Ad(Ad)
        , m_A1(A1)
        , m_I1(I1)
        , m_Ad1(Ad1) { }

    void ConstantMetaData::ToHost(icrar::cpu::MetaData& host)
    {
        host.m_constants = m_constants;

        m_A.ToHost(host.m_A);
        m_I.ToHost(host.m_I);
        m_Ad.ToHost(host.m_Ad);
        m_A1.ToHost(host.m_A1);
        m_I1.ToHost(host.m_I1);
        m_Ad1.ToHost(host.m_Ad1);
    }

    DeviceMetaData::DeviceMetaData(const cpu::MetaData& metadata)
    : m_constantMetadata(std::make_shared<ConstantMetaData>(
        metadata.GetConstants(),
        metadata.GetA(),
        metadata.GetI(),
        metadata.GetAd(),
        metadata.GetA1(),
        metadata.GetI1(),
        metadata.GetAd1()))
    , m_oldUVW(metadata.GetOldUVW())
    , m_UVW(metadata.GetUVW())
    , m_dd(metadata.GetDD())
    , m_direction(metadata.GetDirection())
    , m_avg_data(metadata.GetAvgData())
    {
    }

    DeviceMetaData::DeviceMetaData(std::shared_ptr<ConstantMetaData> constantMetadata, const icrar::cpu::MetaData& metadata)
    : m_constantMetadata(constantMetadata)
    , m_oldUVW(metadata.GetOldUVW())
    , m_UVW(metadata.GetUVW())
    , m_dd(metadata.GetDD())
    , m_direction(metadata.GetDirection())
    , m_avg_data(metadata.GetAvgData())
    {

    }

    const icrar::cpu::Constants& DeviceMetaData::GetConstants() const
    {
        return m_constantMetadata->GetConstants();
    }

    void DeviceMetaData::SetDirection(const icrar::MVDirection& direction)
    {
        m_direction = direction;
    }

    void DeviceMetaData::SetAvgData(int v)
    {
        cudaMemset(m_avg_data.Get(), v, m_avg_data.GetSize());
    }

    void DeviceMetaData::ToHost(const cpu::MetaData& metadata) const
    {
        m_constantMetadata->ToHost(metadata);

        m_oldUVW.ToHost(metadata.m_oldUVW);
        m_UVW.ToHost(metadata.m_UVW);
        metadata.m_direction = m_direction;
        metadata.m_dd = m_dd;
        m_avg_data.ToHost(metadata.m_avg_data);
    }

    void DeviceMetaData::AvgDataToHost(const Eigen::MatrixXcd& host) const
    {
        m_avg_data.ToHost(host);
    }

    cpu::MetaData DeviceMetaData::ToHost() const
    {
        //TODO: tidy up using a constructor for now
        //TODO: casacore::MVuvw and casacore::MVDirection not safe to copy to cuda
        std::vector<icrar::MVuvw> uvwTemp;
        m_UVW.ToHost(uvwTemp);
        cpu::MetaData result = cpu::MetaData();
        ToHost(result);
        return result;
    }

    void DeviceMetaData::ToHostAsync(const cpu::MetaData& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}