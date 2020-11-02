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

#pragma once

#include <icrar/leap-accelerate/model/casa/CalibrateResult.h>
#include <icrar/leap-accelerate/ms/MeasurementSet.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/MVuvw.h>
#include <icrar/leap-accelerate/common/MVDirection.h>
#include <icrar/leap-accelerate/common/Tensor3X.h>
#include <icrar/leap-accelerate/common/vector_extensions.h>
#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/math_conversion.h>

#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <boost/optional.hpp>
#include <boost/noncopyable.hpp>

#include <queue>
#include <vector>
#include <array>
#include <complex>

namespace icrar
{
namespace cpu
{
    /**
     * @brief Container of visibilities for integration
     * 
     */
    class IntegrationResult
    {
        MVDirection m_direction;
        int m_integration_number;
        boost::optional<std::vector<casacore::Vector<double>>> m_data;

    public:
        IntegrationResult(
            icrar::MVDirection direction,
            int integration_number,
            boost::optional<std::vector<casacore::Vector<double>>> data)
            : m_direction(direction)
            , m_integration_number(integration_number)
            , m_data(data)
        {

        }
    };

    /**
     * @brief Container of station calibrations for a given direction.
     * 
     */
    class CalibrationResult
    {
        MVDirection m_direction; // direction the calibration is configured to.
        std::vector<casacore::Matrix<double>> m_stationCalibrations; // Calibrations for each station.

    public:
        /**
         * @brief Construct a new Calibration Result data object
         * 
         * @param direction 
         * @param stationCalibrations 
         */
        CalibrationResult(
            const MVDirection& direction,
            const std::vector<casacore::Matrix<double>>& stationCalibrations)
            : m_direction(direction)
            , m_stationCalibrations(stationCalibrations)
        {
        }

        /**
         * @brief Gets the direction of the calibration
         * 
         * @return const MVDirection 
         */
        const MVDirection GetDirection() const { return m_direction; }
        const std::vector<casacore::Matrix<double>>& GetStationCalibrations() const { return m_stationCalibrations; }

        /**
         * @brief serializes the current object to the provided stream in JSON format.
         * 
         * @param os the output stream
         */
        void Serialize(std::ostream& os) const;

    private:
        template<typename Writer>
        void CreateJsonStrFormat(Writer& writer) const
        {
            assert(m_stationCalibrations.size() == 1);
            assert(m_stationCalibrations[0].shape()[1] == 1);

            writer.StartObject();
            writer.String("direction");
            writer.StartArray();
            for(auto& v : icrar::ToPolar(m_direction))
            {
                writer.Double(v);
            }
            writer.EndArray();

            writer.String("data");
            writer.StartArray();
            for(auto& v : m_stationCalibrations[0])
            {
                writer.Double(v);
            }
            writer.EndArray();

            writer.EndObject();
        }
    };

    using CalibrateResult = std::pair<
        std::vector<std::vector<cpu::IntegrationResult>>,
        std::vector<std::vector<cpu::CalibrationResult>>
    >;

    /**
     * @brief Converts a calibration from casalib containers to eigen3 containers 
     * 
     * @param result 
     * @return icrar::cpu::CalibrateResult 
     */
    icrar::cpu::CalibrateResult ToCalibrateResult(icrar::casalib::CalibrateResult& result);

    void PrintResult(const CalibrateResult& result);
}
}
