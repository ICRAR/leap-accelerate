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

#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <eigen3/Eigen/Core>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace icrar
{
    struct MetaDataCuda
    {
        bool init;

        //int nantennas;
        //int nbaseline;
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

        
    };

    struct Buffers
    {
        std::vector<casacore::MVuvw> oldUVW;

        Eigen::Matrix<std::complex<double>, -1, -1> avg_data; // casacore::Array<casacore::MVuvw> avg_data;
        Eigen::Matrix<double, -1, -1> dd;


        Eigen::Matrix<double, -1, -1> A;
        Eigen::Matrix<double, -1, -1> Ad;
        Eigen::Matrix<double, -1, -1> Ad1;

        Eigen::VectorXf I1;
        Eigen::VectorXf I;
    };
}
