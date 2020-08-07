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

#include <icrar/leap-accelerate/MetaData.h>

#include <icrar/leap-accelerate/cuda/device_vector.h>
#include <icrar/leap-accelerate/cuda/device_matrix.h>

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <eigen3/Eigen/Core>

#include <boost/optional.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace icrar
{
namespace cuda
{
    struct Constants
    {
        int nantennas;
        //int nbaselines;
        int channels; // The number of channels of the current observation
        int num_pols; // The number of polarizations used by the current observation
        int stations; // The number of stations used by the current observation
        int rows;

        double freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz
        std::vector<double> channel_wavelength;

        union
        {
            std::array<double, 2> phase_centre;
            struct
            {
                double phase_centre_ra_rad;
                double phase_centre_dec_rad;
            };
        };

        union
        {
            std::array<double, 2> dlm;
            struct
            {
                double dlm_ra;
                double dlm_dec;
            };
        };

        bool operator==(const Constants& rhs) const;
    };

    class MetaDataCudaHost
    {
        MetaDataCudaHost();

    public:

        bool init = false; //set to true after rotateVisibilities
        Constants m_constants;

        std::vector<casacore::MVuvw> oldUVW;

        Eigen::MatrixXcd avg_data; // casacore::Array<casacore::MVuvw> avg_data;
        Eigen::Matrix3d dd;

        Eigen::MatrixXd A;
        Eigen::VectorXi I;
        Eigen::MatrixXd Ad;

        Eigen::MatrixXd A1;
        Eigen::VectorXi I1;
        Eigen::MatrixXd Ad1;

        MetaDataCudaHost(const MetaData& metadata);

        const Constants& GetConstants() const;

        void CalcUVW(std::vector<casacore::MVuvw>& uvws);
        void SetDD(const casacore::MVDirection& direction);
        void SetWv();

        bool operator==(const MetaDataCudaHost& rhs) const;
    };

    class MetaDataCudaDevice
    {
        MetaDataCudaDevice();
    public:
        bool init;
        Constants constants;

        icrar::cuda::device_vector<casacore::MVuvw> oldUVW;

        icrar::cuda::device_matrix<std::complex<double>> avg_data; // casacore::Array<casacore::MVuvw> avg_data;
        icrar::cuda::device_matrix<double> dd;

        icrar::cuda::device_matrix<double> A;
        icrar::cuda::device_vector<int> I;
        icrar::cuda::device_matrix<double> Ad;
        
        icrar::cuda::device_matrix<double> A1;
        icrar::cuda::device_vector<int> I1;
        icrar::cuda::device_matrix<double> Ad1;

        MetaDataCudaDevice(const MetaDataCudaHost& metadata);

        void ToHost(MetaDataCudaHost& host) const;
        MetaDataCudaHost ToHost() const;
        void ToHostAsync(MetaDataCudaHost& host) const;
    };
}
}
