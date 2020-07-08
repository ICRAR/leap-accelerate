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

#include <casacore/casa/Quanta/MVuvw.h>

using namespace casacore;

namespace icrar
{
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