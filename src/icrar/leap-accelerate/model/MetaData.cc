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

#include <icrar/leap-accelerate/common/constants.h>
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>

#include <icrar/leap-accelerate/algorithm/casa/PhaseRotate.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>
#include <casacore/casa/Quanta/MVuvw.h>

using namespace casacore;

namespace icrar
{
namespace casalib
{
    MetaData::MetaData()
    : A()
    , Ad()
    , I()
    , A1()
    , Ad1()
    , I1()
    {
        
    }

    MetaData::MetaData(const icrar::MeasurementSet& ms)
    {
        auto pms = ms.GetMS();
        auto msc = ms.GetMSColumns();
        auto msmc = ms.GetMSMainColumns();

        this->m_initialized = false;
        this->stations = 0;
        this->nantennas = 0;
        this->solution_interval = 3601;

        this->rows = pms->polarization().nrow();
        this->num_pols = 0;
        if(pms->polarization().nrow() > 0)
        {
            this->num_pols = msc->polarization().numCorr().get(0);
        }

        this->channels = 0;
        this->freq_start_hz = 0;
        this->freq_inc_hz = 0;
        if(pms->spectralWindow().nrow() > 0)
        {
            this->channels = msc->spectralWindow().numChan().get(0);
            this->freq_start_hz = msc->spectralWindow().refFrequency().get(0);
            this->freq_inc_hz = msc->spectralWindow().chanWidth().get(0)(IPosition(1,0));
        }
        this->stations = pms->antenna().nrow();
        if(pms->nrow() > 0)
        {
            auto time_inc_sec = msc->interval().get(0);
        }

        this->phase_centre_ra_rad = 0;
        this->phase_centre_dec_rad = 0;
        if(pms->field().nrow() > 0)
        {
            Vector<MDirection> dir;
            msc->field().phaseDirMeasCol().get(0, dir, true);
            if(dir.size() > 0)
            {
                Vector<double> v = dir(0).getAngle().getValue();
                this->phase_centre_ra_rad = v(0);
                this->phase_centre_dec_rad = v(1);
            }
        }

        Vector<double> range(2, 0.0);
        if(msc->observation().nrow() > 0)
        {
            msc->observation().timeRange().get(0, range);
        }

        casacore::Vector<double> time = msmc->time().getColumn();

        //select the first epoch only
        double epoch = time[0];
        int nEpochs = 0;
        for(int i = 0; i < time.size(); i++)
        {
            if(time[i] == time[0]) nEpochs++;
        }
        auto epochIndices = Slice(0, nEpochs, 1); //TODO assuming epoch indices are sorted
        casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices); 
        casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);

        //Start calculations
        casacore::Matrix<double> A1;
        casacore::Array<std::int32_t> I1;
        std::tie(A1, I1) = icrar::casalib::PhaseMatrixFunction(a1, a2, 0);
        casacore::Matrix<double> Ad1 = icrar::casalib::PseudoInverse(A1);

        casacore::Matrix<double> A;
        casacore::Array<std::int32_t> I;
        std::tie(A, I) = icrar::casalib::PhaseMatrixFunction(a1, a2, -1);
        casacore::Matrix<double> Ad = icrar::casalib::PseudoInverse(A);

        this->A = A;
        this->Ad = Ad;
        this->I = I;
        this->A1 = A1;
        this->Ad1 = Ad1;
        this->I1 = I1;
    }

    MetaData::MetaData(std::istream& input)
    {
        throw std::runtime_error("not implemented");
    }

    void MetaData::CalcUVW(std::vector<casacore::MVuvw>& uvws)
    {
        if(!dd.is_initialized())
        {
            throw std::logic_error("dd must be initialized before using CalcUVW");
        }
        oldUVW = uvws;
        auto size = uvws.size();
        uvws.clear();
        for(int n = 0; n < size; n++)
        {
            auto uvw = icrar::Dot(oldUVW[n], dd.value());
            uvws.push_back(uvw);
        }
    }

    // TODO: rename to CalcDD or UpdateDD
    void MetaData::SetDD(const MVDirection& direction)
    {
        if(!dd.is_initialized())
        {
            dd.reset(casacore::Matrix<double>(3,3));
        }

        auto& dd3d = dd.value();
        dlm_ra = direction.get()[0] - phase_centre_ra_rad;
        dlm_dec = direction.get()[1] - phase_centre_dec_rad;

        dd3d(0,0) = std::cos(dlm_ra) * std::cos(dlm_dec);
        dd3d(0,1) = -std::sin(dlm_ra);
        dd3d(0,2) = std::cos(dlm_ra) * std::sin(dlm_dec);
        
        dd3d(1,0) = std::sin(dlm_ra) * std::cos(dlm_dec);
        dd3d(1,1) = std::cos(dlm_ra);
        dd3d(1,2) = std::sin(dlm_ra) * std::sin(dlm_dec);

        dd3d(2,0) = -std::sin(dlm_dec);
        dd3d(2,1) = 0;
        dd3d(2,2) = std::cos(dlm_dec);
    }

    /**
     * @brief Set the wavelength from meta data
     * TODO: rename to CalcWv or UpdateWv
     * 
     * @param metadata 
     */
    void MetaData::SetWv()
    {
        channel_wavelength = range(
            freq_start_hz,
            freq_start_hz + freq_inc_hz * channels,
            freq_inc_hz);
        
        for(double& v : channel_wavelength)
        {
            v = speed_of_light / v;
        }
    }

    bool MetaData::operator==(const MetaData& rhs) const
    {
        return m_initialized == rhs.m_initialized
        && nantennas == rhs.nantennas
        //&& nbaseline == rhs.nbaseline
        && channels == rhs.channels
        && num_pols == rhs.num_pols
        && stations == rhs.stations
        && rows == rhs.rows
        && freq_start_hz == rhs.freq_start_hz
        && freq_inc_hz == rhs.freq_inc_hz
        && solution_interval == rhs.solution_interval
        && channel_wavelength == rhs.channel_wavelength
        && phase_centre_ra_rad == rhs.phase_centre_ra_rad
        && phase_centre_dec_rad == rhs.phase_centre_dec_rad
        && dlm_ra == rhs.dlm_ra
        && dlm_dec == rhs.dlm_dec
        && oldUVW == rhs.oldUVW
        && icrar::Equal(avg_data, rhs.avg_data)
        && icrar::Equal(dd, rhs.dd)
        && icrar::Equal(A, rhs.A)
        && icrar::Equal(I, rhs.I)
        && icrar::Equal(Ad, rhs.Ad)
        && icrar::Equal(A1, rhs.A1)
        && icrar::Equal(I1, rhs.I1)
        && icrar::Equal(Ad1, rhs.Ad1);

    }
}
}