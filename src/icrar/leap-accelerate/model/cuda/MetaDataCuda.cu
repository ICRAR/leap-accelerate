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

#include "MetaDataCuda.h"
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>


namespace icrar
{
namespace cuda
{
    bool Constants::operator==(const Constants& rhs) const
    {
        return nantennas == rhs.nantennas
        && channels == rhs.channels
        && num_pols == rhs.num_pols
        && stations == rhs.stations
        && rows == rhs.rows
        && solution_interval == rhs.solution_interval
        && freq_start_hz == rhs.freq_start_hz
        && freq_inc_hz == rhs.freq_inc_hz
        && channel_wavelength == rhs.channel_wavelength
        && phase_centre_ra_rad == rhs.phase_centre_ra_rad
        && phase_centre_dec_rad == rhs.phase_centre_dec_rad
        && dlm_ra == rhs.dlm_ra
        && dlm_dec == rhs.dlm_dec;
    }

    MetaData::MetaData(const casalib::MetaData& metadata)
    {
        m_constants.nantennas = metadata.nantennas;
        m_constants.channels = metadata.channels;
        m_constants.num_pols = metadata.num_pols;
        m_constants.stations = metadata.stations;
        m_constants.rows = metadata.rows;
        m_constants.solution_interval = metadata.solution_interval;
        m_constants.freq_start_hz = metadata.freq_start_hz;
        m_constants.freq_inc_hz = metadata.freq_inc_hz;
        m_constants.phase_centre_ra_rad = metadata.phase_centre_ra_rad;
        m_constants.phase_centre_dec_rad = metadata.phase_centre_dec_rad;
        m_constants.dlm_ra = metadata.dlm_ra;
        m_constants.dlm_dec = metadata.dlm_dec;

        if( metadata.channel_wavelength.empty())
        {
            throw std::runtime_error("channel_wavelength: metadata not initialized, use alternative constructor");
        }
        m_constants.channel_wavelength = metadata.channel_wavelength;

        oldUVW = ToUVW(metadata.oldUVW);

        A = ToMatrix(metadata.A);
        I = ToMatrix<int>(metadata.I);
        Ad = ToMatrix(metadata.Ad);

        A1 = ToMatrix(metadata.A1);
        I1 = ToMatrix<int>(metadata.I1);
        Ad1 = ToMatrix(metadata.Ad1);

        if(metadata.dd.is_initialized())
        {
            dd = ToMatrix3x3(metadata.dd.value());
        }
        else
        {
            throw std::runtime_error("dd: metadata not initialized, use alternative constructor");
        }

        if(metadata.avg_data.is_initialized())
        {
            avg_data = ToMatrix(metadata.avg_data.value());
        }
        else
        {
            throw std::runtime_error("avg_data: metadata not initialized, use alternative constructor");
        }
    }

    MetaData::MetaData(const casalib::MetaData& metadata, const casacore::MVDirection& direction, const std::vector<casacore::MVuvw>& uvws)
    {
        m_constants.nantennas = metadata.nantennas;
        m_constants.channels = metadata.channels;
        m_constants.num_pols = metadata.num_pols;
        m_constants.stations = metadata.stations;
        m_constants.rows = metadata.rows;
        m_constants.solution_interval = metadata.solution_interval;
        m_constants.freq_start_hz = metadata.freq_start_hz;
        m_constants.freq_inc_hz = metadata.freq_inc_hz;
        m_constants.phase_centre_ra_rad = metadata.phase_centre_ra_rad;
        m_constants.phase_centre_dec_rad = metadata.phase_centre_dec_rad;
        m_constants.dlm_ra = metadata.dlm_ra;
        m_constants.dlm_dec = metadata.dlm_dec;

        A = ToMatrix(metadata.A);
        I = ToMatrix<int>(metadata.I);
        Ad = ToMatrix(metadata.Ad);

        A1 = ToMatrix(metadata.A1);
        I1 = ToMatrix<int>(metadata.I1);
        Ad1 = ToMatrix(metadata.Ad1);

        SetWv();
        SetDD(direction);
        CalcUVW(ToUVW(uvws));
        assert(UVW.size() == uvws.size());

        if(metadata.avg_data.is_initialized())
        {
            avg_data = ToMatrix(metadata.avg_data.value());
        }
    }

    const Constants& MetaData::GetConstants() const
    {
        return m_constants;
    }

    void MetaData::CalcUVW(const std::vector<icrar::MVuvw>& uvws)
    {
        this->oldUVW = uvws;
        auto size = uvws.size();
        this->UVW = std::vector<icrar::MVuvw>();
        this->UVW.reserve(uvws.size());
        for(int n = 0; n < size; n++)
        {
            UVW.push_back(uvws[n] * dd);
        }

        avg_data = Eigen::MatrixXcd::Zero(UVW.size(), m_constants.num_pols);
    }

    void MetaData::SetDD(const casacore::MVDirection& direction)
    {
        this->direction = ToUVW(direction);

        m_constants.dlm_ra = direction.get()[0] - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = direction.get()[1] - m_constants.phase_centre_dec_rad;

        dd = Eigen::Matrix3d();
        dd(0,0) = std::cos(m_constants.dlm_ra) * std::cos(m_constants.dlm_dec);
        dd(0,1) = -std::sin(m_constants.dlm_ra);
        dd(0,2) = std::cos(m_constants.dlm_ra) * std::sin(m_constants.dlm_dec);
        
        dd(1,0) = std::sin(m_constants.dlm_ra) * std::cos(m_constants.dlm_dec);
        dd(1,1) = std::cos(m_constants.dlm_ra);
        dd(1,2) = std::sin(m_constants.dlm_ra) * std::sin(m_constants.dlm_dec);

        dd(2,0) = -std::sin(m_constants.dlm_dec);
        dd(2,1) = 0;
        dd(2,2) = std::cos(m_constants.dlm_dec);
    }

    void MetaData::SetWv()
    {
        m_constants.channel_wavelength = range(
            m_constants.freq_start_hz,
            m_constants.freq_start_hz + m_constants.freq_inc_hz * m_constants.channels,
            m_constants.freq_inc_hz);
        
        double speed_of_light = 299792458.0;
        for(double& v : m_constants.channel_wavelength)
        {
            v = speed_of_light / v;
        }
    }

    bool MetaData::operator==(const MetaData& rhs) const
    {
        return m_constants == rhs.m_constants
        && oldUVW == rhs.oldUVW
        && UVW == rhs.UVW
        && A == rhs.A
        && I == rhs.I
        && Ad == rhs.Ad
        && A1 == rhs.A1
        && I1 == rhs.I1
        && Ad1 == rhs.Ad1
        && dd == rhs.dd
        && avg_data == rhs.avg_data;
    }

    DeviceMetaData::DeviceMetaData(const MetaData& metadata)
    : constants(metadata.GetConstants())
    , UVW(metadata.UVW)
    , oldUVW(metadata.oldUVW)
    , dd(metadata.dd)
    , avg_data(metadata.avg_data)
    , A(metadata.A)
    , I(metadata.I)
    , Ad(metadata.Ad)
    , A1(metadata.A1)
    , I1(metadata.I1)
    , Ad1(metadata.Ad1)
    {

    }

    void DeviceMetaData::ToHost(MetaData& metadata) const
    {
        metadata.m_constants = constants;

        A.ToHost(metadata.A);
        I.ToHost(metadata.I);
        Ad.ToHost(metadata.Ad);
        A1.ToHost(metadata.A1);
        I1.ToHost(metadata.I1);
        Ad1.ToHost(metadata.Ad1);

        oldUVW.ToHost(metadata.oldUVW);
        UVW.ToHost(metadata.UVW);
        metadata.direction = direction;
        metadata.dd = dd;
        avg_data.ToHost(metadata.avg_data);
    }

    MetaData DeviceMetaData::ToHost() const
    {
        //TODO: tidy up using a constructor for now
        //TODO: casacore::MVuvw and casacore::MVDirection not safe to copy to cuda
        std::vector<icrar::MVuvw> uvwTemp;
        UVW.ToHost(uvwTemp);
        MetaData result = MetaData(casalib::MetaData(), casacore::MVDirection(), ToCasaUVW(uvwTemp));
        ToHost(result);
        return result;
    }

    void DeviceMetaData::ToHostAsync(MetaData& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}