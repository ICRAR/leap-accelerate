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

#pragma once

#include <icrar/leap-accelerate/model/cpu/calibration/BeamCalibration.h>
#include <icrar/leap-accelerate/math/math_conversion.h>
#include <icrar/leap-accelerate/exception/exception.h>

#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#ifndef __NVCC__
#include <rapidjson/document.h>
#endif // __NVCC__
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>

#include <vector>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Contains a single calibration solution.
     * 
     */
    class Calibration
    {
        double m_startEpoch;
        double m_endEpoch;
        std::vector<BeamCalibration> m_beamCalibrations;

    public:
        /**
         * @brief Creates an empty calibration
         * 
         * @param startEpoch 
         * @param endEpoch 
         */
        Calibration(double startEpoch, double endEpoch);

        Calibration(double startEpoch, double endEpoch, std::vector<cpu::BeamCalibration>&& beamCalibrations);

        double GetStartEpoch() const;
        
        double GetEndEpoch() const;

        bool IsApprox(const Calibration& calibration, double tolerence);

        const std::vector<BeamCalibration>& GetBeamCalibrations() const;

        std::vector<BeamCalibration>& GetBeamCalibrations();

        void Serialize(std::ostream& os, bool pretty = false) const;

        template<typename Writer>
        void Write(Writer& writer) const
        {
            writer.StartObject();
                writer.String("epoch"); writer.StartObject();
                    writer.String("start"); writer.Double(m_startEpoch);
                    writer.String("end"); writer.Double(m_endEpoch);
                writer.EndObject();
                writer.String("calibration"); writer.StartArray();
                    for(auto& beamCalibration : m_beamCalibrations)
                    {
                        beamCalibration.Write(writer);
                    }
                writer.EndArray();
            writer.EndObject();
        }

        static Calibration Parse(std::istream& is);
        static Calibration Parse(const std::string& json);
#ifndef __NVCC__
        static Calibration Parse(const rapidjson::Value& doc);
#endif // __NVCC__
    };
}
}
