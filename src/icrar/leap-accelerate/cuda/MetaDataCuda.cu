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
        && freq_start_hz == rhs.freq_start_hz
        && freq_inc_hz == rhs.freq_inc_hz
        && channel_wavelength == rhs.channel_wavelength
        && phase_centre_ra_rad == rhs.phase_centre_ra_rad
        && phase_centre_dec_rad == rhs.phase_centre_dec_rad
        && dlm_ra == rhs.dlm_ra
        && dlm_dec == rhs.dlm_dec;
    }

    MetaDataCudaHost::MetaDataCudaHost(const MetaData& metadata)
    {
        m_constants.nantennas = metadata.nantennas;
        m_constants.channels = metadata.channels;
        m_constants.num_pols = metadata.num_pols;
        m_constants.stations = metadata.stations;
        m_constants.rows = metadata.rows;
        m_constants.freq_start_hz = metadata.freq_start_hz;
        m_constants.freq_inc_hz = metadata.freq_inc_hz;
        m_constants.phase_centre_ra_rad = metadata.phase_centre_ra_rad;
        m_constants.phase_centre_dec_rad = metadata.phase_centre_dec_rad;
        m_constants.dlm_ra = metadata.dlm_ra;
        m_constants.dlm_dec = metadata.dlm_dec;

        init = metadata.init;
        oldUVW = metadata.oldUVW;

        avg_data = ConvertMatrix(metadata.avg_data);
        
        if(metadata.dd.is_initialized())
        {
            dd = ConvertMatrix3x3(metadata.dd.value());
        }
        else
        {
            throw std::runtime_error("dd is required");
        }

        A = ConvertMatrix(metadata.A);
        I = ConvertMatrix<int>(metadata.I);
        Ad = ConvertMatrix(metadata.Ad);

        A1 = ConvertMatrix(metadata.A1);
        I1 = ConvertMatrix<int>(metadata.I1);
        Ad1 = ConvertMatrix(metadata.Ad1);
    }

    const Constants& MetaDataCudaHost::GetConstants() const
    {
        return m_constants;
    }

    void MetaDataCudaHost::CalcUVW(std::vector<casacore::MVuvw>& uvws)
    {
        this->oldUVW = uvws;
        auto size = uvws.size();
        uvws.clear();
        for(int n = 0; n < size; n++)
        {
            auto uvw = icrar::Dot(uvws[n], dd);
            uvws.push_back(uvw);
        }
    }

    void MetaDataCudaHost::SetDD(const casacore::MVDirection& direction)
    {
        m_constants.dlm_ra = direction.get()[0] - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = direction.get()[1] - m_constants.phase_centre_dec_rad;

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

    void MetaDataCudaHost::SetWv()
    {
        double speed_of_light = 299792458.0;
        m_constants.channel_wavelength = range(
            m_constants.freq_start_hz,
            m_constants.freq_inc_hz,
            m_constants.freq_start_hz + m_constants.freq_inc_hz * m_constants.channels);
        for(double& v : m_constants.channel_wavelength)
        {
            v = speed_of_light / v;
        }
    }

    bool MetaDataCudaHost::operator==(const MetaDataCudaHost& rhs) const
    {
        return init == rhs.init
        && m_constants == rhs.m_constants
        && oldUVW == rhs.oldUVW
        && avg_data == rhs.avg_data
        && dd == rhs.dd
        && A == rhs.A
        && I == rhs.I
        && Ad == rhs.Ad
        && A1 == rhs.A1
        && I1 == rhs.I1
        && Ad1 == rhs.Ad1;
    }

    MetaDataCudaDevice::MetaDataCudaDevice(const MetaDataCudaHost& metadata)
    : constants(metadata.GetConstants())
    , init(metadata.init)
    , oldUVW(metadata.oldUVW)
    , avg_data(metadata.avg_data)
    , dd(metadata.dd)
    , A(metadata.A)
    , I(metadata.I)
    , Ad(metadata.Ad)
    , A1(metadata.A1)
    , I1(metadata.I1)
    , Ad1(metadata.Ad1)
    {

    }

    void MetaDataCudaDevice::ToHost(MetaDataCudaHost& metadata) const
    {
        metadata.init = init;
        metadata.m_constants = constants;
        oldUVW.ToHost(metadata.oldUVW);
        avg_data.ToHost(metadata.avg_data);
        dd.ToHost(metadata.dd);
        A.ToHost(metadata.A);
        I.ToHost(metadata.I);
        Ad.ToHost(metadata.Ad);
        A1.ToHost(metadata.A1);
        I1.ToHost(metadata.I1);
        Ad1.ToHost(metadata.Ad1);
    }

    MetaDataCudaHost MetaDataCudaDevice::ToHost() const
    {
        auto meta = MetaData();
        meta.SetDD(casacore::MVDirection(0.0, 0.0));
        MetaDataCudaHost result = MetaDataCudaHost(meta);
        ToHost(result);
        return result;
    }

    void MetaDataCudaDevice::ToHostAsync(MetaDataCudaHost& host) const
    {
        throw std::runtime_error("not implemented");
    }
}
}