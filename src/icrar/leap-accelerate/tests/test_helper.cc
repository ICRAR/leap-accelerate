/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "test_helper.h"
#include <icrar/leap-accelerate/common/stream_extensions.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>
#include <icrar/leap-accelerate/tests/math/eigen_helper.h>

void assert_near_metadata(const icrar::cpu::LeapData& expected, const icrar::cpu::LeapData& actual, const std::string& ln, const std::string& rn, const std::string& file, int line)
{
    if(expected != actual)
    {
        std::cerr << ln << " != " << rn << "\n";
        std::cerr << "LeapData not exactly equal at " << file << ":" << line << std::endl;
    }

    const double THRESHOLD = 0.001;
    
    ASSERT_EQ(expected.GetConstants().nbaselines, actual.GetConstants().nbaselines);
    ASSERT_EQ(expected.GetConstants().channels, actual.GetConstants().channels);
    ASSERT_EQ(expected.GetConstants().num_pols, actual.GetConstants().num_pols);
    ASSERT_EQ(expected.GetConstants().stations, actual.GetConstants().stations);
    ASSERT_EQ(expected.GetConstants().rows, actual.GetConstants().rows);
    ASSERT_EQ(expected.GetConstants().freq_start_hz, actual.GetConstants().freq_start_hz);
    ASSERT_EQ(expected.GetConstants().freq_inc_hz, actual.GetConstants().freq_inc_hz);
    ASSERT_EQ(expected.GetConstants().phase_centre_ra_rad, actual.GetConstants().phase_centre_ra_rad);
    ASSERT_EQ(expected.GetConstants().phase_centre_dec_rad, actual.GetConstants().phase_centre_dec_rad);
    ASSERT_EQ(expected.GetConstants().dlm_ra, actual.GetConstants().dlm_ra);
    ASSERT_EQ(expected.GetConstants().dlm_dec, actual.GetConstants().dlm_dec);


    ASSERT_MEQD(expected.GetA(), actual.GetA(), THRESHOLD);
    ASSERT_MEQI(expected.GetI(), actual.GetI(), 0);
    ASSERT_MEQD(expected.GetAd(), actual.GetAd(), THRESHOLD);
    ASSERT_MEQD(expected.GetA1(), actual.GetA1(), THRESHOLD);
    ASSERT_MEQI(expected.GetI1(), actual.GetI1(), 0);
    ASSERT_MEQD(expected.GetAd1(), actual.GetAd1(), THRESHOLD);

    ASSERT_MEQ3D(expected.GetDD(), actual.GetDD(), THRESHOLD);
    ASSERT_MEQCD(expected.GetAvgData(), actual.GetAvgData(), THRESHOLD);
    
    // TODO(calgray): ensure these copy correctly
    //ASSERT_EQ(expected.direction, actual.direction);
    //ASSERT_EQ(expected.oldUVW, actual.oldUVW); 
    //ASSERT_EQ(expected.UVW, actual.UVW);

    //ASSERT_EQ(expected, actual);
    //ASSERT_EQ(expectedIntegration, integration);
}