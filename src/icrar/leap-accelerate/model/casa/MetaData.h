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

#include <icrar/leap-accelerate/math/casa/matrix_invert.h>
#include <icrar/leap-accelerate/common/MVDirection.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/optional.hpp>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace icrar
{
    class MeasurementSet;
}

#pragma clang
namespace icrar
{
namespace casalib
{
    struct MetaData
    {
        bool m_initialized;

        int nbaseline;
        int channels; // The number of channels of the current observation
        int num_pols; // The number of polarizations used by the current observation
        int stations; // The number of stations used by the current observation
        int rows;

        double freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz

        std::vector<double> channel_wavelength;
        std::vector<casacore::MVuvw> oldUVW;

        boost::optional<casacore::Matrix<double>> dd;

        boost::optional<casacore::Matrix<std::complex<double>>> avg_data;

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

        casacore::Matrix<double> A; // [baselines+1,stations] Antennas (all, including flagged) PhaseMatrix
        casacore::Matrix<double> Ad; // A inverse
        
        casacore::Matrix<double> A1; // [baselines,stations] Antennas ((all, including flagged) with baseline PhaseMatrix
        casacore::Matrix<double> Ad1; //A1 inverse

        casacore::Vector<int> I1; // The indexes of the antennas used by A
        casacore::Vector<int> I; // The indexes of the antennas used by A1

    public:
        MetaData();
        MetaData(std::istream& input);
        MetaData(const icrar::MeasurementSet& ms);
        
        /**
         * @brief Gets the number of baselines
         * 
         * @return int
         */
        int GetBaselines() const { return stations * (stations + 1) / 2; }

        /**
         * @brief 
         * 
         * @param metadata 
         */
        void SetWv();

        /**
         * @brief 
         * 
         * @param uvw 
         * @param metadata 
         */
        void CalcUVW(std::vector<casacore::MVuvw>& uvw);

        /**
         * @brief 
         * 
         * @param metadata 
         * @param direction 
         */
        void SetDD(const casacore::MVDirection& direction);

        void SetDD(const icrar::MVDirection& direction);

        bool operator==(const MetaData& rhs) const;
    };
} // namespace casalib
} // namespace icrar
