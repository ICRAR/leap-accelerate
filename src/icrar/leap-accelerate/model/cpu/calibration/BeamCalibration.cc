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

#include "BeamCalibration.h"

namespace icrar
{
namespace cpu
{
    BeamCalibration::BeamCalibration(
        SphericalDirection direction,
        Eigen::MatrixXd calibration)
        : m_direction(std::move(direction))
        , m_antennaPhases(std::move(calibration))
    {
    }

    BeamCalibration::BeamCalibration(const std::pair<SphericalDirection, Eigen::MatrixXd>& beamCalibration)
    {
        std::tie(m_direction, m_antennaPhases) = beamCalibration;
    }

    bool BeamCalibration::IsApprox(const BeamCalibration& beamCalibration, double threshold)
    {
        bool equal = m_direction == beamCalibration.m_direction
        && m_antennaPhases.rows() == beamCalibration.m_antennaPhases.rows()
        && m_antennaPhases.cols() == beamCalibration.m_antennaPhases.cols()
        && m_antennaPhases.isApprox(beamCalibration.m_antennaPhases, threshold);
        return equal;
    }

    const SphericalDirection& BeamCalibration::GetDirection() const
    {
        return m_direction;
    }

    const Eigen::VectorXd& BeamCalibration::GetAntennaPhases() const
    {
        return m_antennaPhases;
    }

    void BeamCalibration::Serialize(std::ostream& os, bool pretty) const
    {
        constexpr uint32_t PRECISION = 15;
        os.precision(PRECISION);
        os.setf(std::ios::fixed);

        rapidjson::OStreamWrapper osw = {os};
        if(pretty)
        {
            rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
            Write(writer);
        }
        else
        {
            rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
            Write(writer);
        }
    }

    BeamCalibration BeamCalibration::Parse(const rapidjson::Value& doc)
    {
        if(!doc.IsObject())
        {
            throw icrar::exception("expected a beam calibration", __FILE__, __LINE__);
        }
        const auto& direction = doc["direction"];
        const auto sphericalDirection = SphericalDirection(direction[0].GetDouble(), direction[1].GetDouble());

        const auto& calibrationJson = doc["antennaPhases"];
        if(!calibrationJson.IsArray())
        {
            throw icrar::exception("expected an array", __FILE__, __LINE__);
        }
        Eigen::VectorXd calibrationVector(calibrationJson.Size());
        std::transform(calibrationJson.Begin(), calibrationJson.End(), calibrationVector.begin(),
        [](const rapidjson::Value& v){ return v.GetDouble(); });
        return { sphericalDirection, calibrationVector };
    }
} // namespace cpu
} // namespace icrar
