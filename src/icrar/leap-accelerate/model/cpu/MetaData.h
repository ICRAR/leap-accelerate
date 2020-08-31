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
    class DeviceMetaData;
}
}

namespace icrar
{
namespace cpu
{
    struct Constants
    {
        int nantennas;
        int nbaselines; //the total number station pairs (excluding self cycles) 

        int channels; // The number of channels of the current observation
        int num_pols; // The number of polarizations used by the current observation
        int stations; // The number of stations used by the current observation
        int rows;

        int solution_interval; //TODO can remove?

        double freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz

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

        __device__ __host__ double GetChannelWavelength(int i) const
        {
            return speed_of_light / (freq_inc_hz + i * freq_inc_hz);
        }

        bool operator==(const Constants& rhs) const;
    };

    class MetaData
    {
        MetaData() {}

        Constants m_constants;
        
        Eigen::MatrixXd A;
        Eigen::VectorXi I;
        Eigen::MatrixXd Ad;

        Eigen::MatrixXd A1;
        Eigen::VectorXi I1;
        Eigen::MatrixXd Ad1;

    public:
        std::vector<icrar::MVuvw> oldUVW; // late initialized
        std::vector<icrar::MVuvw> UVW; // late initialized

        icrar::MVDirection direction; // late initialized
        Eigen::Matrix3d dd; // late initialized
        Eigen::MatrixXcd avg_data; // late initialized

        MetaData(const casalib::MetaData& metadata);
        MetaData(const casalib::MetaData& metadata, const casacore::MVDirection& direction, const std::vector<casacore::MVuvw>& uvws);
        
        MetaData(icrar::MeasurementSet& ms);
        MetaData(
            const Constants& constants,
            const double* A, int ARows, int ACols,
            const int* I, int ILength,
            const double* Ad, int AdRows, int AdCols,
            const double* A1, int A1Rows, int A1Cols,
            Eigen::Matrix3d& dd,
            const std::complex<double>* avg_data, int avg_dataRows, int avg_dataCols)
        {
            // TODO
        }

        const Constants& GetConstants() const;

        const Eigen::MatrixXd& GetA() const;
        const Eigen::VectorXi& GetI() const;
        const Eigen::MatrixXd& GetAd() const;

        const Eigen::MatrixXd& GetA1() const;
        const Eigen::VectorXi& GetI1() const;
        const Eigen::MatrixXd& GetAd1() const;

        void CalcUVW(const std::vector<icrar::MVuvw>& uvws);
        void SetDD(const casacore::MVDirection& direction);
        void SetWv();

        bool operator==(const MetaData& rhs) const;

        friend class icrar::cuda::DeviceMetaData;
    };
    }
}
