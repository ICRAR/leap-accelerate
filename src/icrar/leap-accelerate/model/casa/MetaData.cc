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
#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/algorithm/casa/PhaseMatrixFunction.h>

#include <casacore/casa/Quanta/MVDirection.h>
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
    , A1()
    , Ad1()
    , I1()
    , I()
    {
        
    }

    MetaData::MetaData(const icrar::MeasurementSet& ms, double minimumBaselineThreshold)
    : nbaselines(0)
    , channels(0)
    , num_pols(0)
    , stations(0)
    , rows(0)
    , freq_start_hz(0)
    , freq_inc_hz(0)
    , min_baseline_length(minimumBaselineThreshold)
    , phase_centre_ra_rad(0)
    , phase_centre_dec_rad(0)
    {
        auto pms = ms.GetMS();
        auto msc = ms.GetMSColumns();
        auto msmc = ms.GetMSMainColumns();

        this->m_initialized = false;

        this->nbaselines = ms.GetNumBaselines();
        this->rows = ms.GetNumRows();
        this->num_pols = ms.GetNumPols();

        if(pms->spectralWindow().nrow() > 0)
        {
            this->channels = msc->spectralWindow().numChan().get(0);
            this->freq_start_hz = msc->spectralWindow().refFrequency().get(0);
            this->freq_inc_hz = msc->spectralWindow().chanWidth().get(0)(IPosition(1,0));
        }

        this->stations = ms.GetNumStations();
        
        // if(pms->nrow() > 0)
        // {
        //     auto time_inc_sec = msc->interval().get(0);
        // }

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

        //select the first epoch only
        auto epochIndices = Slice(0, ms.GetNumBaselines(), 1); //TODO assuming epoch indices are sorted
        casacore::Vector<std::int32_t> a1 = msmc->antenna1().getColumn()(epochIndices);
        casacore::Vector<std::int32_t> a2 = msmc->antenna2().getColumn()(epochIndices);

        casacore::Vector<bool> baselineFlags = ConvertVector(ms.GetFilteredBaselines(minimumBaselineThreshold));

        if(a1.size() != a2.size())
        {
            throw icrar::file_exception("a1 and a2 not equal size", ms.GetFilepath().is_initialized() ? ms.GetFilepath().get() : "unknown", __FILE__, __LINE__);
        }
        for(size_t i = a2.size(); i < a2.size(); ++i)
        {
            if(a1(i) < 0)
            {
                throw icrar::file_exception("a1 less than 0", ms.GetFilepath().is_initialized() ? ms.GetFilepath().get() : "unknown", __FILE__, __LINE__);
            }
            if(a2(i) < 0)
            {
                throw icrar::file_exception("a2 less than 0", ms.GetFilepath().is_initialized() ? ms.GetFilepath().get() : "unknown", __FILE__, __LINE__);
            }
        }


        //Start calculations
        std::tie(this->A1, this->I1) = icrar::casalib::PhaseMatrixFunction(a1, a2, baselineFlags, 0);
        this->Ad1 = icrar::casalib::PseudoInverse(A1);

        std::tie(this->A, this->I) = icrar::casalib::PhaseMatrixFunction(a1, a2, baselineFlags, -1);
        this->Ad = icrar::casalib::PseudoInverse(A);
    }

    MetaData::MetaData(std::istream& /*input*/, double /*minimumBaselineThreshold*/)
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
        for(size_t n = 0; n < size; n++)
        {
            auto uvw = icrar::Dot(oldUVW[n], dd.value());
            uvws.push_back(uvw);
        }
    }

    // TODO: rename to CalcDD or UpdateDD
    void MetaData::SetDD(const casacore::MVDirection& direction)
    {
        if(!dd.is_initialized())
        {
            dd.reset(casacore::Matrix<double>(3,3));
        }

        auto& dd3d = dd.value();

        //NOTE: using polar direction
        //This is the way using astropy -- we need to repeat
        /*
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        # First move to epoch of Obs
        coords=[(metadata['phase_centre_ra_rad'],metadata['phase_centre_dec_rad'])]
        c_obs_phase_centre=SkyCoord(coords, frame=FK5, unit=(u.rad, u.rad))
                    #,obstime=metadata['observation_date'],location=EarthLocation.of_site('mwa'))
        coords=[(direction[0],direction[1])]
        c_obs_direction=SkyCoord(coords, frame=FK5, unit=(u.rad, u.rad))
                                    #,obstime=metadata['observation_date'],location=EarthLocation.of_site('mwa'))
        offset=c_obs_phase_centre.spherical_offsets_to(c_obs_direction)
        */

        // Also see:
        // https://stackoverflow.com/questions/25404613/converting-spherical-coordinates-to-cartesian

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

    void MetaData::SetDD(const icrar::MVDirection& direction)
    {
        SetDD(ToCasaDirection(direction));
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
        && nbaselines == rhs.nbaselines
        && channels == rhs.channels
        && num_pols == rhs.num_pols
        && stations == rhs.stations
        && rows == rhs.rows
        && freq_start_hz == rhs.freq_start_hz
        && freq_inc_hz == rhs.freq_inc_hz
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