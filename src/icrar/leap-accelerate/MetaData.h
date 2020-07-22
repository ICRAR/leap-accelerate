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

#include <icrar/leap-accelerate/algorithm/cpu/PhaseRotate.h>
#include <icrar/leap-accelerate/math/cpu/Invert.h>

#include <casacore/ms/MeasurementSets.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <complex>

namespace icrar
{
    struct MetaData
    {
        MetaData();
        MetaData(std::istream& input);
        MetaData(const casacore::MeasurementSet& ms);

        bool init;

        int nantennas;
        int nbaseline;
        int channels; // The number of channels of the current observation
        int num_pols; // The number of polarizations used by the current observation
        int stations; // The number of stations used by the current observation
        int rows;

        double freq_start_hz; // The frequency of the first channel, in Hz
        double freq_inc_hz; // The frequency incrmeent between channels, in Hz

        int solution_interval;

        std::vector<double> channel_wavelength;
        std::vector<casacore::MVuvw> oldUVW;

        casacore::Matrix<std::complex<double>> avg_data; // casacore::Array<casacore::MVuvw> avg_data;
        casacore::Matrix<double> dd;

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

        casacore::Matrix<double> A; // Antennas all PhaseMatrix
        casacore::Matrix<double> Ad; // A inverse
        
        casacore::Matrix<double> A1; // Antennas with baseline PhaseMatrix
        casacore::Matrix<double> Ad1; //A1 inverse

        casacore::Array<int> I1;
        casacore::Array<int> I;

        // void SetDlmRa(double value) { dlm_ra; }
        // double GetDlmRa();
        // void SetDlmdDec(double value);
        // double GetDlmdDec();
    };

    class MetaDataVerified
    {
        int m_antennas;
        int m_baseline;
        int m_channels;
        int m_stations;

    public:
        MetaDataVerified(int antennas, int baselines, int channels, int polarizations, int stations)
        : m_antennas(antennas)
        , m_baseline(baselines)
        , m_channels(channels)
        , m_stations(stations)
        {

        }
    };

    /**
     * @brief 
     * 
     * @param metadata 
     * @param direction 
     */
    void SetDD(MetaData& metadata, const casacore::MVDirection& direction);
    
    /**
     * @brief 
     * 
     * @param metadata 
     */
    void SetWv(MetaData& metadata);
    
    /**
     * @brief 
     * 
     * @param uvw 
     * @param metadata 
     */
    void CalcUVW(std::vector<casacore::MVuvw>& uvw, MetaData& metadata);

    //class Stats
    // {
    // public:
    //     bool m_init;
    //     int m_channels; // The number of channels of the current observation
    //     int m_num_pols; // The number of polarizations used by the current observation
    //     int m_stations; // The number of stations used by the current observation
    //     int m_rows;

    //     int solution_interval; // Solve for every 'interval' cycles
    //     double phase_centre_ra_rad; // The RA phase centre in radians
    //     double phase_centre_dec_rad; // The DEC phase centre in radians

    //     Matrixd A;
    //     Matrixd Ad;
    //     Matrixd Ad1;
    //     Matrixd I1;
    //     Matrixd I;
    // };
}
