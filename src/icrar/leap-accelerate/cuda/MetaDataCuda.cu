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
    MetaDataCudaHost::MetaDataCudaHost(MetaData& metadata)
    {
        
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
            auto uvw = icrar::Dot(uvws[n], this->dd);
            uvws.push_back(uvw);
        }
    }

    void MetaDataCudaHost::SetDD(const casacore::MVDirection& direction)
    {
        m_constants.dlm_ra = direction.get()[0] - m_constants.phase_centre_ra_rad;
        m_constants.dlm_dec = direction.get()[1] - m_constants.phase_centre_dec_rad;

        this->dd(0,0) = cos(m_constants.dlm_ra) * cos(m_constants.dlm_dec);
        this->dd(0,1) = -sin(m_constants.dlm_ra);
        this->dd(0,2) = cos(m_constants.dlm_ra) * sin(m_constants.dlm_dec);
        
        this->dd(1,0) = sin(m_constants.dlm_ra) * cos(m_constants.dlm_dec);
        this->dd(1,1) = cos(m_constants.dlm_ra);
        this->dd(1,2) = sin(m_constants.dlm_ra) * sin(m_constants.dlm_dec);

        this->dd(2,0) = -sin(m_constants.dlm_dec);
        this->dd(2,1) = 0;
        this->dd(2,2) = cos(m_constants.dlm_dec);
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
}
}