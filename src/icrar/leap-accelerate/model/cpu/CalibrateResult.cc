/**
*    ICRAR - International Centre for Radio Astronomy Research
*    (c) UWA - The University of Western Australia
*    Copyright by UWA (in the framework of the ICRAR)
*    All rights reserved
*
*    This library is free software; you can redistribute it and/or
*    modify it under the terms of the GNU Lesser General Public
*    License as published by the Free Software Foundation; either
*    version 2.1 of the License, or (at your option) any later version.
*
*    This library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*    Lesser General Public License for more details.
*
*    You should have received a copy of the GNU Lesser General Public
*    License along with this library; if not, write to the Free Software
*    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*    MA 02111-1307  USA
*/

#include <icrar/leap-accelerate/model/cpu/CalibrateResult.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/math/vector_extensions.h>

#include <rapidjson/stringbuffer.h>
#include <rapidjson/prettywriter.h>

namespace icrar
{
namespace cpu
{
    void CalibrationResult::Serialize(std::ostream& os) const
    {
        constexpr uint32_t PRECISION = 15;
        os.precision(PRECISION);
        os.setf(std::ios::fixed);

        rapidjson::StringBuffer s;

        //TODO(calgray): could also support PrettyWriter
        rapidjson::Writer<rapidjson::StringBuffer> writer(s);
        CreateJsonStrFormat(writer);
        os << s.GetString() << std::endl;
    }

    void PrintResult(const CalibrateResult& result, std::ostream& out)
    {
        for(auto& calibrations : result.second)
        {
            for(auto& calibration : calibrations)
            {
                calibration.Serialize(out);
            }
        }
    }
} // namespace cpu
} // namespace icrar
