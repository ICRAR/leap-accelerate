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

    MetaDataPortable::MetaDataPortable(const MetaData& metadata)
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

        oldUVW = metadata.oldUVW;

        A = ConvertMatrix(metadata.A);
        I = ConvertMatrix<int>(metadata.I);
        Ad = ConvertMatrix(metadata.Ad);

        A1 = ConvertMatrix(metadata.A1);
        I1 = ConvertMatrix<int>(metadata.I1);
        Ad1 = ConvertMatrix(metadata.Ad1);

        if(metadata.dd.is_initialized())
        {
            dd = ConvertMatrix3x3(metadata.dd.value());
        }
        else
        {
            throw std::runtime_error("dd: metadata not initialized, use alternative constructor");
        }

        if(metadata.avg_data.is_initialized())
        {
            avg_data = ConvertMatrix(metadata.avg_data.value());
        }
        else
        {
            throw std::runtime_error("avg_data: metadata not initialized, use alternative constructor");
        }
    }

    MetaDataPortable::MetaDataPortable(const MetaData& metadata, const casacore::MVDirection& direction, const std::vector<casacore::MVuvw>& uvws)
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

        A = ConvertMatrix(metadata.A);
        I = ConvertMatrix<int>(metadata.I);
        Ad = ConvertMatrix(metadata.Ad);

        A1 = ConvertMatrix(metadata.A1);
        I1 = ConvertMatrix<int>(metadata.I1);
        Ad1 = ConvertMatrix(metadata.Ad1);

        SetWv();
        SetDD(direction);
        CalcUVW(uvws);

        if(metadata.avg_data.is_initialized())
        {
            avg_data = ConvertMatrix(metadata.avg_data.value());
        }
    }

    const Constants& MetaDataPortable::GetConstants() const
    {
        return m_constants;
    }

    void MetaDataPortable::CalcUVW(const std::vector<casacore::MVuvw>& uvws)
    {
        this->oldUVW = uvws;
        auto size = uvws.size();
        this->UVW = std::vector<casacore::MVuvw>();
        for(int n = 0; n < size; n++)
        {
            auto uvw = icrar::Dot(uvws[n], dd);
            UVW.push_back(uvw);
        }

        avg_data = Eigen::MatrixXcd::Zero(UVW.size(), m_constants.num_pols);
    }

    void MetaDataPortable::SetDD(const casacore::MVDirection& direction)
    {
        this->direction = direction;

        m_constants.dlm_ra = direction.get()[0] - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = direction.get()[1] - m_constants.phase_centre_dec_rad;

        dd = Eigen::Matrix3d();
        dd(0,0) = cos(m_constants.dlm_ra) * cos(m_constants.dlm_dec);
        dd(0,1) = -sin(m_constants.dlm_ra);
        dd(0,2) = cos(m_constants.dlm_ra) * sin(m_constants.dlm_dec);
        
        dd(1,0) = sin(m_constants.dlm_ra) * cos(m_constants.dlm_dec);
        dd(1,1) = cos(m_constants.dlm_ra);
        dd(1,2) = sin(m_constants.dlm_ra) * sin(m_constants.dlm_dec);

        dd(2,0) = -sin(m_constants.dlm_dec);
        dd(2,1) = 0;
        dd(2,2) = cos(m_constants.dlm_dec);
    }

    void MetaDataPortable::SetWv()
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

    bool MetaDataPortable::operator==(const MetaDataPortable& rhs) const
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

    MetaDataCudaDevice::MetaDataCudaDevice(const MetaDataPortable& metadata)
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

    void MetaDataCudaDevice::ToHost(MetaDataPortable& metadata) const
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
        //dd.ToHost(metadata.dd);
        metadata.dd = dd;
        avg_data.ToHost(metadata.avg_data);
    }

    MetaDataPortable MetaDataCudaDevice::ToHost() const
    {
        //TODO: tidy up using a constructor for now
        //TODO: casacore::MVuvw and casacore::MVDirection not safe to copy to cuda
        std::vector<casacore::MVuvw> uvwTemp;
        UVW.ToHost(uvwTemp);
        MetaDataPortable result = MetaDataPortable(MetaData(), direction, uvwTemp);
        ToHost(result);
        return result;
    }

    void MetaDataCudaDevice::ToHostAsync(MetaDataPortable& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}