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

namespace icrar
{
namespace casalib
{
    struct MetaData
    {
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        bool m_initialized;

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        int nbaseline;
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        int channels; // The number of channels of the current observation
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        int num_pols; // The number of polarizations used by the current observation
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        int stations; // The number of stations used by the current observation
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        int rows;

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        double freq_start_hz; // The frequency of the first channel, in Hz
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        std::vector<double> channel_wavelength;
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        std::vector<casacore::MVuvw> oldUVW;
        
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        boost::optional<casacore::Matrix<double>> dd;
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        boost::optional<casacore::Matrix<std::complex<double>>> avg_data;

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        double phase_centre_ra_rad;
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        double phase_centre_dec_rad;

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        double dlm_ra;
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        double dlm_dec;

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        casacore::Matrix<double> A; // [baselines+1,stations] Antennas (all, including flagged) PhaseMatrix
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        casacore::Matrix<double> Ad; // A inverse
        
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        casacore::Matrix<double> A1; // [baselines,stations] Antennas ((all, including flagged) with baseline PhaseMatrix
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        casacore::Matrix<double> Ad1; //A1 inverse

        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        casacore::Vector<int> I1; // The indexes of the antennas used by A
        // NOLINTNEXTLINE(misc-non-private-member-variables-in-classes)
        casacore::Vector<int> I; // The indexes of the antennas used by A1

    public:
        MetaData();
        explicit MetaData(std::istream& input);
        explicit MetaData(const icrar::MeasurementSet& ms);
        
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
        void CalcUVW(std::vector<casacore::MVuvw>& uvw); // NOLINT(google-runtime-references)

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
