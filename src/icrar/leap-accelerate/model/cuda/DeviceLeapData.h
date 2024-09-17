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

#pragma once

#ifdef CUDA_ENABLED

#include <icrar/leap-accelerate/common/SphericalDirection.h>

#include <icrar/leap-accelerate/common/constants.h>
#include <icrar/leap-accelerate/model/cpu/LeapData.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>


#include <Eigen/Core>

#include <boost/optional.hpp>

#include <memory>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

namespace icrar
{
namespace cuda
{
    /**
     * Container class of uniform gpu buffers available to all cuda
     * threads that are const/immutable per calibration.
     */
    class ConstantBuffer
    {
        icrar::cpu::Constants m_constants;
        
        device_matrix<double> m_A;
        device_vector<int> m_I;
        device_matrix<double> m_Ad;

        device_matrix<double> m_A1;
        device_vector<int> m_I1;
        device_matrix<double> m_Ad1;

    public:
        /**
         * @brief Construct a new Constant Buffer object
         * 
         * @param constants 
         * @param A 
         * @param I 
         * @param Ad 
         * @param A1 
         * @param I1 
         * @param Ad1 
         */
        ConstantBuffer(
            const icrar::cpu::Constants& constants,
            device_matrix<double>&& A,
            device_vector<int>&& I,
            device_matrix<double>&& Ad,
            device_matrix<double>&& A1,
            device_vector<int>&& I1,
            device_matrix<double>&& Ad1);

        const icrar::cpu::Constants& GetConstants() const { return m_constants; }
        const device_matrix<double>& GetA() const { return m_A; } 
        const device_vector<int>& GetI() const { return m_I; }
        const device_matrix<double>& GetAd() const { return m_Ad; }
        const device_matrix<double>& GetA1() const { return m_A1; }
        const device_vector<int>& GetI1() const { return m_I1; }
        const device_matrix<double>& GetAd1() const { return m_Ad1; }

        void ToHost(icrar::cpu::LeapData& host) const;
        void ToHostAsync(icrar::cpu::LeapData& host) const;
    };

    /**
     * @brief LeapData Variables allocated per direction
     * 
     */
    class DirectionBuffer
    {
        // TODO(calgray) use device types
        SphericalDirection m_direction;
        Eigen::Matrix3d m_dd;

        device_matrix<std::complex<double>> m_avgData;

    public:
        /**
         * @brief Constructs a new Direction Buffer object initializing all memory
         * 
         * @param direction 
         * @param dd 
         * @param avgData 
         */
        DirectionBuffer(
            SphericalDirection direction,
            Eigen::Matrix3d dd,
            const Eigen::MatrixXcd& avgData);

        /**
         * @brief Constructs a new Direction Buffer object for late initialization
         *  
         * @param avgDataRows 
         * @param avgDataCols 
         */
        DirectionBuffer(
            int avgDataRows,
            int avgDataCols);

        const SphericalDirection& GetDirection() const { return m_direction; }
        const Eigen::Matrix3d& GetDD() const { return m_dd; }

        device_matrix<std::complex<double>>& GetAvgData() { return m_avgData; }

        void SetDirection(const SphericalDirection& direction);
        void SetDD(const Eigen::Matrix3d& dd);
    };

    /**
     * Represents the complete collection of LeapData that
     * resides on the GPU for leap-calibration
     */
    class DeviceLeapData
    {
        std::shared_ptr<ConstantBuffer> m_constantBuffer; // Constant buffer, never null
        std::shared_ptr<DirectionBuffer> m_directionBuffer;

    public:
        DeviceLeapData(DeviceLeapData&& other) noexcept = default;
        DeviceLeapData& operator=(DeviceLeapData&& other) noexcept = default;

        /**
         * @brief Construct a new Device LeapData object from the equivalent object on CPU memory. This copies to
         * all device buffers
         * 
         * @param leapData 
         */
        explicit DeviceLeapData(const icrar::cpu::LeapData& leapData);
        
        /**
         * @brief Construct a new Device LeapData object from the equivalent object on CPU memory. This copies to
         * all device buffers
         * 
         * @param constantBuffer 
         * @param directionBuffer 
         */
        DeviceLeapData(
            std::shared_ptr<ConstantBuffer> constantBuffer,
            std::shared_ptr<DirectionBuffer> directionBuffer);

        const icrar::cpu::Constants& GetConstants() const;

        const SphericalDirection& GetDirection() const { return m_directionBuffer->GetDirection(); }
        const Eigen::Matrix3d& GetDD() const { return m_directionBuffer->GetDD(); }
        
        const ConstantBuffer& GetConstantBuffer() const { return *m_constantBuffer; }
        const device_matrix<std::complex<double>>& GetAvgData() const { return m_directionBuffer->GetAvgData(); };
        device_matrix<std::complex<double>>& GetAvgData() { return m_directionBuffer->GetAvgData(); };

        void SetAvgData(int v);

        void ToHost(icrar::cpu::LeapData& host) const;
        icrar::cpu::LeapData ToHost() const;
        
        void ToHostAsync(icrar::cpu::LeapData& host) const;
    };
}
}

#endif // CUDA_ENABLED