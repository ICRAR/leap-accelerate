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

#include "DeviceLeapData.h"
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/exception/exception.h>
#include <icrar/leap-accelerate/math/cpu/matrix_invert.h>
#include <icrar/leap-accelerate/math/cuda/matrix_invert.h>

namespace icrar
{
namespace cuda
{
    ConstantBuffer::ConstantBuffer(
            const icrar::cpu::Constants& constants,
            device_matrix<double>&& A,
            device_vector<int>&& I,
            device_matrix<double>&& Ad,
            device_matrix<double>&& A1,
            device_vector<int>&& I1,
            device_matrix<double>&& Ad1)
        : m_constants(constants)
        , m_A(std::move(A))
        , m_I(std::move(I))
        , m_Ad(std::move(Ad))
        , m_A1(std::move(A1))
        , m_I1(std::move(I1))
        , m_Ad1(std::move(Ad1))
        { }

    void ConstantBuffer::ToHost(icrar::cpu::LeapData& host) const
    {
        host.m_constants = m_constants;

        m_A.ToHost(host.m_A);
        m_I.ToHost(host.m_I);
        m_Ad.ToHost(host.m_Ad);
        m_A1.ToHost(host.m_A1);
        m_I1.ToHost(host.m_I1);
        m_Ad1.ToHost(host.m_Ad1);
    }

    void ConstantBuffer::ToHostAsync(icrar::cpu::LeapData& host) const
    {
        host.m_constants = m_constants;

        m_A.ToHostAsync(host.m_A);
        m_I.ToHostAsync(host.m_I);
        m_Ad.ToHostAsync(host.m_Ad);
        m_A1.ToHostAsync(host.m_A1);
        m_I1.ToHostAsync(host.m_I1);
        m_Ad1.ToHostAsync(host.m_Ad1);
    }

    DirectionBuffer::DirectionBuffer(
        SphericalDirection direction,
        Eigen::Matrix3d dd,
        const Eigen::MatrixXcd& avgData)
    : m_direction(std::move(direction))
    , m_dd(std::move(dd))
    , m_avgData(avgData)
    {}

    DirectionBuffer::DirectionBuffer(
        int avgDataRows,
        int avgDataCols)
    : m_avgData(avgDataRows, avgDataCols)
    {}

    void DirectionBuffer::SetDirection(const SphericalDirection& direction)
    {
        m_direction = direction;
    }

    void DirectionBuffer::SetDD(const Eigen::Matrix3d& dd)
    {
        m_dd = dd;
    }

    DeviceLeapData::DeviceLeapData(const cpu::LeapData& leapData)
    : m_constantBuffer(std::make_shared<ConstantBuffer>(
        leapData.GetConstants(),
        device_matrix<double>(leapData.GetA()),
        device_vector<int>(leapData.GetI()),
        device_matrix<double>(leapData.GetAd()),
        device_matrix<double>(leapData.GetA1()),
        device_vector<int>(leapData.GetI1()),
        device_matrix<double>(leapData.GetAd1())))
    , m_directionBuffer(std::make_shared<DirectionBuffer>(
        leapData.GetDirection(),
        leapData.GetDD(),
        leapData.GetAvgData()))
    {}

    DeviceLeapData::DeviceLeapData(
        std::shared_ptr<ConstantBuffer> constantBuffer,
        std::shared_ptr<DirectionBuffer> directionBuffer)
    : m_constantBuffer(std::move(constantBuffer))
    , m_directionBuffer(std::move(directionBuffer))
    {}

    const icrar::cpu::Constants& DeviceLeapData::GetConstants() const
    {
        return m_constantBuffer->GetConstants();
    }

    void DeviceLeapData::SetAvgData(int v)
    {
        cudaMemset(m_directionBuffer->GetAvgData().Get(), v, m_directionBuffer->GetAvgData().GetSize());
    }

    void DeviceLeapData::ToHost(cpu::LeapData& leapData) const
    {
        m_constantBuffer->ToHost(leapData);
        leapData.m_direction = m_directionBuffer->GetDirection();
        leapData.m_dd = m_directionBuffer->GetDD();
        m_directionBuffer->GetAvgData().ToHost(leapData.m_avgData);
    }

    cpu::LeapData DeviceLeapData::ToHost() const
    {
        cpu::LeapData result = cpu::LeapData();
        ToHost(result);
        return result;
    }

    void DeviceLeapData::ToHostAsync(cpu::LeapData& leapData) const
    {
        m_constantBuffer->ToHostAsync(leapData);

        leapData.m_direction = m_directionBuffer->GetDirection();
        leapData.m_dd = m_directionBuffer->GetDD();
        m_directionBuffer->GetAvgData().ToHostVectorAsync(leapData.m_avgData);
    }
} // namespace cuda
} // namespace icrar
#endif // CUDA_ENABLED
