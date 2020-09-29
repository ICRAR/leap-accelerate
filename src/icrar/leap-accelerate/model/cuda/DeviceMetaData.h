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
#include <icrar/leap-accelerate/model/casa/MetaData.h>
#include <icrar/leap-accelerate/model/cpu/MetaData.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <Eigen/Core>

#include <boost/optional.hpp>

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
     * threads and are immutable.
     */
    class UniformMetaData
    {
    public:
        icrar::cpu::Constants constants;
        
        icrar::cuda::device_matrix<double> A;
        icrar::cuda::device_vector<int> I;
        icrar::cuda::device_matrix<double> Ad;
        
        icrar::cuda::device_matrix<double> A1;
        icrar::cuda::device_vector<int> I1;
        icrar::cuda::device_matrix<double> Ad1;
    };

    /**
     * Represents the complete collection of MetaData that
     * resides on the GPU for leap-calibration
     */
    class DeviceMetaData
    {
        DeviceMetaData();

        icrar::cpu::Constants constants;
    public:
        
        icrar::cuda::device_matrix<double> A;
        icrar::cuda::device_vector<int> I;
        icrar::cuda::device_matrix<double> Ad;
        
        icrar::cuda::device_matrix<double> A1;
        icrar::cuda::device_vector<int> I1;
        icrar::cuda::device_matrix<double> Ad1;


        // Metadata that is zero'd before execution

        icrar::cuda::device_vector<icrar::MVuvw> oldUVW;
        icrar::cuda::device_vector<icrar::MVuvw> UVW;
        icrar::MVDirection direction;
        Eigen::Matrix3d dd;
        icrar::cuda::device_matrix<std::complex<double>> avg_data;

    public:
        DeviceMetaData(const icrar::cpu::MetaData& metadata);

        const icrar::cpu::Constants& GetConstants();

        void ToHost(icrar::cpu::MetaData& host) const;
        icrar::cpu::MetaData ToHost() const;
        void ToHostAsync(icrar::cpu::MetaData& host) const;
    };
}
}
