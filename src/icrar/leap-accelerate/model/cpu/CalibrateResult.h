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
    class IntegrationResult
    {
        int m_integrationNumber;
        MVDirection m_direction;
        boost::optional<std::vector<Eigen::VectorXd>> m_data;

    public:
        IntegrationResult(
            int integrationNumber,
            icrar::MVDirection direction,
            boost::optional<std::vector<Eigen::VectorXd>> data)
            : m_integrationNumber(integrationNumber)
            , m_direction(std::move(direction))
            , m_data(std::move(data))
        {

        }
    };

    class CalibrationResult
    {
        MVDirection m_direction;
        Eigen::MatrixXd m_data;

    public:
        CalibrationResult(
            MVDirection direction,
            Eigen::MatrixXd data)
            : m_direction(std::move(direction))
            , m_data(std::move(data))
        {
        }

        /**
         * @brief Gets the calibration direction
         * 
         * @return const MVDirection 
         */
        const MVDirection GetDirection() const { return m_direction; }

        /**
         * @brief Get the calibration Vector for the antenna array in the specified direction
         * 
         * @return const Eigen::MatrixXd 
         */
        const Eigen::MatrixXd& GetData() const { return m_data; }

        void Serialize(std::ostream& os) const;

    private:
        template<typename Writer>
        void CreateJsonStrFormat(Writer& writer) const
        {
            assert(m_data.cols() == 1);

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
            for(int i = 0; i < m_data.rows(); ++i)
            {
                writer.Double(m_data(i,0));
            }
            writer.EndArray();

            writer.EndObject();
        }
    };

    using CalibrateResult = std::pair<
        std::vector<std::vector<cpu::IntegrationResult>>,
        std::vector<std::vector<cpu::CalibrationResult>>
    >;

    icrar::cpu::CalibrateResult ToCalibrateResult(icrar::casalib::CalibrateResult& result);

    void PrintResult(const CalibrateResult& result, std::ostream& out);
}
}
