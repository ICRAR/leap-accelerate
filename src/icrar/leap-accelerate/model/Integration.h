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

#include <icrar/leap-accelerate/common/Tensor3X.h>

#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <icrar/leap-accelerate/common/eigen_3_3_beta_1_2_support.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <boost/optional.hpp>

#include <vector>
#include <array>
#include <complex>


namespace icrar
{
    class MeasurementSet;

    class Integration
    {
    public:
        //Integration();
        Integration(const icrar::MeasurementSet& ms, int integrationNumber, int channels, int baselines, int polarizations, int uvws);

        Eigen::Tensor<std::complex<double>, 3> data; //data is an array data[nch][nbl][npol]

        std::vector<casacore::MVuvw> uvw; //uvw is an array uvw[3][nbl]
        int integration_number;

        union
        {
            std::array<int, 4> parameters; // index, 0, channels, baselines
            struct
            {
                size_t index;
                size_t x;
                size_t channels;
                size_t baselines;
            };
        };

        bool operator==(const Integration& rhs) const;
    };

    class IntegrationResult
    {
        casacore::MVDirection m_direction;
        int m_integration_number;
        boost::optional<std::vector<casacore::Array<double>>> m_data;

    public:
        IntegrationResult(
            casacore::MVDirection direction,
            int integration_number,
            boost::optional<std::vector<casacore::Array<double>>> data)
            : m_direction(direction)
            , m_integration_number(integration_number)
            , m_data(data)
        {

        }
    };

    class CalibrationResult
    {
        casacore::MVDirection m_direction;
        std::vector<casacore::Matrix<double>> m_data;

    public:
        CalibrationResult(
            const casacore::MVDirection& direction,
            const std::vector<casacore::Matrix<double>>& data)
            : m_direction(direction)
            , m_data(data)
        {
        }

        const casacore::MVDirection GetDirection() const { return m_direction; }
        const std::vector<casacore::Matrix<double>>& GetData() const { return m_data; }

        //bool operator==(const CalibrationResult& rhs) const;
    };
}