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
        this->constants.dlm_ra = direction.get()[0] - this->constants.phase_centre_ra_rad;
        this->constants.dlm_dec = direction.get()[1] - this->constants.phase_centre_dec_rad;

        this->dd(0,0) = cos(this->constants.dlm_ra) * cos(this->constants.dlm_dec);
        this->dd(0,1) = -sin(this->constants.dlm_ra);
        this->dd(0,2) = cos(this->constants.dlm_ra) * sin(this->constants.dlm_dec);
        
        this->dd(1,0) = sin(this->constants.dlm_ra) * cos(this->constants.dlm_dec);
        this->dd(1,1) = cos(this->constants.dlm_ra);
        this->dd(1,2) = sin(this->constants.dlm_ra) * sin(this->constants.dlm_dec);

        this->dd(2,0) = -sin(this->constants.dlm_dec);
        this->dd(2,1) = 0;
        this->dd(2,2) = cos(this->constants.dlm_dec);
    }

    void MetaDataCudaHost::SetWv()
    {
        double speed_of_light = 299792458.0;
        this->constants.channel_wavelength = range(
            this->constants.freq_start_hz,
            this->constants.freq_inc_hz,
            this->constants.freq_start_hz + this->constants.freq_inc_hz * this->constants.channels);
        for(double& v : this->constants.channel_wavelength)
        {
            v = speed_of_light / v;
        }
    }
}
}