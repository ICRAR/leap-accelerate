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
    ConstantBuffer::ConstantBuffer(
            const icrar::cpu::Constants& constants,
            const Eigen::MatrixXd& A,
            const Eigen::VectorXi& I,
            const Eigen::MatrixXd& Ad,
            const Eigen::MatrixXd& A1,
            const Eigen::VectorXi& I1,
            const Eigen::MatrixXd& Ad1)
        : m_constants(constants)
        , m_A(A)
        , m_I(I)
        , m_Ad(Ad)
        , m_A1(A1)
        , m_I1(I1)
        , m_Ad1(Ad1) { }

    void ConstantBuffer::ToHost(icrar::cpu::MetaData& host) const
    {
        host.m_constants = m_constants;

        m_A.ToHost(host.m_A);
        m_I.ToHost(host.m_I);
        m_Ad.ToHost(host.m_Ad);
        m_A1.ToHost(host.m_A1);
        m_I1.ToHost(host.m_I1);
        m_Ad1.ToHost(host.m_Ad1);
    }

    // SolutionIntervalBuffer::SolutionIntervalBuffer(const Eigen::MatrixXd& oldUvw)
    // : m_oldUVW(oldUvw)
    // {}

    SolutionIntervalBuffer::SolutionIntervalBuffer(const std::vector<icrar::MVuvw>& oldUvw)
    : m_oldUVW(oldUvw)
    {}

    DirectionBuffer::DirectionBuffer(
        const icrar::MVDirection& direction,
        const Eigen::Matrix3d& dd,
        const std::vector<icrar::MVuvw>& uvw,
        const Eigen::MatrixXcd& avgData)
    : m_direction(direction)
    , m_dd(dd)
    , m_UVW(uvw)
    , m_avgData(avgData)
    {}

    DirectionBuffer::DirectionBuffer(
        const icrar::MVDirection& direction,
        const Eigen::Matrix3d& dd,
        int uvwRows,
        int avgDataRows,
        int avgDataCols)
    : m_direction(direction)
    , m_dd(dd)
    , m_UVW(uvwRows)
    , m_avgData(avgDataRows, avgDataCols)
    {}

    void DirectionBuffer::SetDirection(const icrar::MVDirection& direction)
    {
        m_direction = direction;
    }

    void DirectionBuffer::SetDD(const Eigen::Matrix3d& dd)
    {
        m_dd = dd;
    }

    DeviceMetaData::DeviceMetaData(const cpu::MetaData& metadata)
    : m_constantBuffer(std::make_shared<ConstantBuffer>(
        metadata.GetConstants(),
        metadata.GetA(),
        metadata.GetI(),
        metadata.GetAd(),
        metadata.GetA1(),
        metadata.GetI1(),
        metadata.GetAd1()))
    , m_solutionIntervalBuffer(std::make_shared<SolutionIntervalBuffer>(
        metadata.GetOldUVW()))
    , m_directionBuffer(std::make_shared<DirectionBuffer>(
        metadata.GetDirection(),
        metadata.GetDD(),
        metadata.GetUVW(),
        metadata.GetAvgData()))
    {}

    DeviceMetaData::DeviceMetaData(
        std::shared_ptr<ConstantBuffer> constantBuffer,
        std::shared_ptr<SolutionIntervalBuffer> SolutionIntervalBuffer,
        std::shared_ptr<DirectionBuffer> directionBuffer)
    : m_constantBuffer(constantBuffer)
    , m_solutionIntervalBuffer(SolutionIntervalBuffer)
    , m_directionBuffer(directionBuffer)
    {}

    const icrar::cpu::Constants& DeviceMetaData::GetConstants() const
    {
        return m_constantBuffer->GetConstants();
    }

    void DeviceMetaData::SetAvgData(int v)
    {
        cudaMemset(m_directionBuffer->m_avgData.Get(), v, m_directionBuffer->m_avgData.GetSize());
    }

    void DeviceMetaData::ToHost(cpu::MetaData& metadata) const
    {
        m_constantBuffer->ToHost(metadata);

        m_solutionIntervalBuffer->GetOldUVW().ToHost(metadata.m_oldUVW);
        m_directionBuffer->m_UVW.ToHost(metadata.m_UVW);
        metadata.m_direction = m_directionBuffer->m_direction;
        metadata.m_dd = m_directionBuffer->m_dd;
        m_directionBuffer->m_avgData.ToHost(metadata.m_avgData);
    }

    void DeviceMetaData::AvgDataToHost(Eigen::MatrixXcd& host) const
    {
        m_directionBuffer->m_avgData.ToHost(host);
    }

    cpu::MetaData DeviceMetaData::ToHost() const
    {
        //TODO: tidy up using a constructor for now
        //TODO: casacore::MVuvw and casacore::MVDirection not safe to copy to cuda
        std::vector<icrar::MVuvw> uvwTemp;
        m_directionBuffer->m_UVW.ToHost(uvwTemp);
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