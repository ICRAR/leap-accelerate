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

#include "MetaData.h"

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>
#include <casacore/casa/Quanta/MVuvw.h>

using namespace casacore;

namespace icrar
{
    MetaData::MetaData()
    {
        
    }

    MetaData::MetaData(const casacore::MeasurementSet& ms)
    {
        //See https://github.com/OxfordSKA/OSKAR/blob/master/oskar/ms/src/oskar_ms_open.cpp

        //casacore::MeasurementSet& cms = const_cast<casacore::MeasurementSet&>(ms);

        //auto cms = casacore::MeasurementSet();

        void* pms = (void*)&ms;
        casacore::MSColumns& msc = *(casacore::MSColumns*)pms;
        
        // = casacore::MSColumns(cms); // NOTE: only xenial casacore is not const qualified
        casacore::MSMainColumns& msmc = *(casacore::MSMainColumns*)pms;
        // = casacore::MSMainColumns(cms); // NOTE: only xenial casacore is not const qualified

        this->init = true;
        this->stations = 0;
        this->nantennas = 0;
        this->solution_interval = 3601;

        this->rows = ms.polarization().nrow();
        this->num_pols = 0;
        if(ms.polarization().nrow() > 0)
        {
            this->num_pols = msc.polarization().numCorr().get(0);
        }

        this->channels = 0;
        this->freq_start_hz = 0;
        this->freq_inc_hz = 0;
        if(ms.spectralWindow().nrow() > 0)
        {
            this->channels = msc.spectralWindow().numChan().get(0);
            this->freq_start_hz = msc.spectralWindow().refFrequency().get(0);
            this->freq_inc_hz = msc.spectralWindow().chanWidth().get(0)(IPosition(1,0));
        }
        this->stations = ms.antenna().nrow();
        if(ms.nrow() > 0)
        {
            auto time_inc_sec = msc.interval().get(0);
        }

        this->phase_centre_ra_rad = 0;
        this->phase_centre_dec_rad = 0;
        if(ms.field().nrow() > 0)
        {
            Vector<MDirection> dir;
            msc.field().phaseDirMeasCol().get(0, dir, true);
            if(dir.size() > 0)
            {
                Vector<double> v = dir(0).getAngle().getValue();
                this->phase_centre_ra_rad = v(0);
                this->phase_centre_dec_rad = v(1);
            }
        }

        Vector<double> range(2, 0.0);
        if(msc.observation().nrow() > 0)
        {
            msc.observation().timeRange().get(0, range);
        }
        //start_time = range[0];
        //end_time = range[1];


        casacore::Vector<double> time = msmc.time().getColumn();
        //msmc.time();

        this->nantennas = 4853; //TODO
        casacore::Vector<std::int32_t> a1 = msmc.antenna1().getColumn()(Slice(0, 4853, 1)); //TODO
        casacore::Vector<std::int32_t> a2 = msmc.antenna2().getColumn()(Slice(0, 4853, 1)); //TODO

        //Start calculations
        casacore::Matrix<double> A1;
        casacore::Array<std::int32_t> I1;
        std::tie(A1, I1) = icrar::cpu::PhaseMatrixFunction(a1, a2, 0);
        casacore::Matrix<double> Ad1 = icrar::cpu::PseudoInverse(A1);

        casacore::Matrix<double> A;
        casacore::Array<std::int32_t> I;
        std::tie(A, I) = icrar::cpu::PhaseMatrixFunction(a1, a2, -1);
        casacore::Matrix<double> Ad = icrar::cpu::PseudoInverse(A);

        this->A = A;
        this->Ad = Ad;
        this->A1 = A1;
        this->Ad1 = Ad1;
        this->I1 = I1;
        this->I = I;
    }

    MetaData::MetaData(std::istream& input)
    {
        throw std::runtime_error("not implemented");
    }

    void CalcUVW(std::vector<MVuvw>& uvws, MetaData& metadata)
    {
        metadata.oldUVW = uvws;
        auto size = uvws.size();
        uvws.clear();
        for(int n = 0; n < size; n++)
        {
            auto uvw = icrar::Dot(uvws[n], metadata.dd);
            uvws.push_back(uvw);
        }
    }

    /**
     * @brief 
     * TODO: rename to CalcDD or UpdateDD
     * @param metadata 
     * @param direction 
     */
    void SetDD(MetaData& metadata, const MVDirection& direction)
    {
        metadata.dlm_ra = direction.get()[0] - metadata.phase_centre_ra_rad;
        metadata.dlm_dec = direction.get()[1] - metadata.phase_centre_dec_rad;

        metadata.dd(IPosition(0,0)) = cos(metadata.dlm_ra) * cos(metadata.dlm_dec);
        metadata.dd(IPosition(0,1)) = -sin(metadata.dlm_ra);
        metadata.dd(IPosition(0,2)) = cos(metadata.dlm_ra) * sin(metadata.dlm_dec);
        
        metadata.dd(IPosition(1,0)) = sin(metadata.dlm_ra) * cos(metadata.dlm_dec);
        metadata.dd(IPosition(1,1)) = cos(metadata.dlm_ra);
        metadata.dd(IPosition(1,2)) = sin(metadata.dlm_ra) * sin(metadata.dlm_dec);

        metadata.dd(IPosition(2,0)) = -sin(metadata.dlm_dec);
        metadata.dd(IPosition(2,1)) = 0;
        metadata.dd(IPosition(2,2)) = cos(metadata.dlm_dec);
    }

    /**
     * @brief Set the wavelength from meta data
     * TODO: rename to CalcWv or UpdateWv
     * 
     * @param metadata 
     */
    void SetWv(MetaData& metadata)
    {
        double speed_of_light = 299792458.0;
        metadata.channel_wavelength = range(
            metadata.freq_start_hz,
            metadata.freq_inc_hz,
            metadata.freq_start_hz + metadata.freq_inc_hz * metadata.channels);
        for(double& v : metadata.channel_wavelength)
        {
            v = speed_of_light / v;
        }
    }
}