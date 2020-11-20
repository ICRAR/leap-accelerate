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

#pragma once

#include <icrar/leap-accelerate/common/MVuvw.h>
#include <icrar/leap-accelerate/common/MVDirection.h>

#include <icrar/leap-accelerate/common/constants.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <Eigen/Core>

#include <boost/optional.hpp>

#include <memory>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

#include <cuComplex.h>

namespace icrar
{
namespace cuda
{
    /**
     * Container of uniform gpu buffers available to all cuda
     * threads that are const/immutable per calibration.
     */
    class ConstantMetaData
    {
        icrar::cpu::Constants m_constants;
        
        device_matrix<double> m_A;
        device_vector<int> m_I;
        device_matrix<double> m_Ad;

        device_matrix<double> m_A1;
        device_vector<int> m_I1;
        device_matrix<double> m_Ad1;

    public:
        ConstantMetaData(
            const icrar::cpu::Constants constants,
            Eigen::MatrixXd A,
            Eigen::VectorXi I,
            Eigen::MatrixXd Ad,
            Eigen::MatrixXd A1,
            Eigen::VectorXi I1,
            Eigen::MatrixXd Ad1);

        const icrar::cpu::Constants& GetConstants() const { return m_constants; }
        const device_matrix<double>& GetA() const { return m_A; } 
        const device_vector<int>& GetI() const { return m_I; }
        const device_matrix<double>& GetAd() const { return m_Ad; }
        const device_matrix<double>& GetA1() const { return m_A1; }
        const device_vector<int>& GetI1() const { return m_I1; }
        const device_matrix<double>& GetAd1() const { return m_Ad1; }

        void ToHost(icrar::cpu::MetaData& host);
    };

    /**
     * Represents the complete collection of MetaData that
     * resides on the GPU for leap-calibration
     */
    class DeviceMetaData
    {
        DeviceMetaData();

        std::shared_ptr<ConstantMetaData> m_constantMetadata; // Constant buffer, never null

        // Metadata that is zero'd before execution
        device_vector<icrar::MVuvw> m_oldUVW;
        device_vector<icrar::MVuvw> m_UVW;
        icrar::MVDirection m_direction;
        Eigen::Matrix3d m_dd;
        device_matrix<std::complex<double>> m_avg_data;

    public:
        /**
         * @brief Construct a new Device MetaData object from the equivalent object on CPU memory. This copies to
         * all device buffers
         * 
         * @param metadata 
         */
        [[deprecated]]
        DeviceMetaData(const icrar::cpu::MetaData& metadata);
        
        /**
         * @brief Construct a new Device MetaData object from the equivalent object on CPU memory. This copies to
         * all device buffers
         * 
         * @param metadata 
         */
        DeviceMetaData(std::shared_ptr<ConstantMetaData> uniformMetadata, const icrar::cpu::MetaData& metadata);

        const icrar::cpu::Constants& GetConstants() const;

        const device_vector<icrar::MVuvw>& GetOldUVW() const { return m_oldUVW; }
        const device_vector<icrar::MVuvw>& GetUVW() const { return m_UVW; }
        const icrar::MVDirection& GetDirection() const { return m_direction; }
        const Eigen::Matrix3d& GetDD() const { return m_dd; }
        const device_matrix<std::complex<double>>& GetAvgData() { return m_avg_data; };

        void SetDirection(const icrar::MVDirection& direction);
        void SetAvgData(int v);

        void ToHost(icrar::cpu::MetaData& host) const;
        icrar::cpu::MetaData ToHost() const;
        void ToHostAsync(icrar::cpu::MetaData& host) const;

        /**
         * @brief Copies average data to host memory
         * 
         * @param host 
         */
        void AvgDataToHost(Eigen::MatrixXcd& host) const;
    };
}
}
