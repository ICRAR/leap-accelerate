
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

#include "PhaseRotate.h"
#include "icrar/leap-accelerate/wsclean/chgcentre.h"

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <filesystem>
#include <optional>
#include <exception>
#include <memory>

using namespace casacore;
using Matrixd = Matrix<double>;
using Matrixi = Matrix<int>;

namespace icrar
{
    class Stats
    {
        bool m_init;
        int m_channels; // The number of channels of the current observation
        int m_num_pols; // The number of polarizations used by the current observation
        int m_stations; // The number of stations used by the current observation
        int m_rows;
        double m_freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz
        int solution_interval; // Solve for every 'interval' cycles
        double phase_centre_ra_rad; // The RA phase centre in radians
        double phase_centre_dec_rad; // The DEC phase centre in radians
        Matrixd A;
        Matrixd Ad;
        Matrixd Ad1;
        Matrixd I1;
        Matrixd I;
    };

    class MetaData
    {
        bool init;
        int oldUVW;
        int nantennas;
        int nbaseline;
        int nchannels;

        std::array<double, 2> dd;
        std::array<double, 2> dlm;
    };

    class Integration
    {
        void* data;
        std::array<double, 3> uvw;
        std::array<int, 4> parameters; // index, 0, channels, baselines
    };

    /**
     *Given the antenna lists from MS and (optionally) RefAnt & Map:
     * If non-negative RefAnt is provided it only forms the matrix for baselines with that antenna.
     * If True Map is provided it returns the index map for the matrix (only useful if RefAnt set).
     *
     * This function generates and returns the linear matrix for the phase calibration (only)
     */
    Matrixd PhaseMatrixFuntion(Array a1, Array a2, int refAnt=-1, bool map=false)
    {
        //TODO array equal
        //int nAnt = 1 + (a1 == a2) ? 1 : 0;
        int nAnt = 2;
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        Matrixd A = Matrixd(a1.size() + 1, a1.max());
        A.fill(0);

        Matrixi I = Matrixi(a1.size() + 1);
        I.fill(1);

        int k = 0;

        for(auto it : a1)
        {
            
        }
    }

    /**
     * Rotate the visibilities of the provided integrations
     */
    void RotateVisibilities(Integration& integration, MetaData& metadata, const MDirection& direction)
    {

    }
    
    void PhaseRotate(casacore::MeasurementSet& ms, std::vector<MDirection> directions)
    {
        MSAntenna antenna = ms.antenna();
    }
}