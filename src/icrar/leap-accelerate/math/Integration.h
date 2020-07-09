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

#include <casacore/casa/Quanta/MVuvw.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <vector>
#include <array>
#include <complex>


namespace icrar
{
    class Integration
    {
    public:
        std::vector<std::vector<std::complex<double>>> data;
        std::vector<casacore::MVuvw> uvw;

        union
        {
            std::array<int, 4> parameters; // index, 0, channels, baselines
            struct
            {
                int index;
                int x;
                int channels;
                int baselines;
            };
        };
    };

    class IntegrationResult
    {
        casacore::MVDirection m_direction;
        int m_integration_number;
        std::vector<std::vector<std::complex<double>>> m_data;

    public:
        IntegrationResult(
            casacore::MVDirection direction,
            int integration_number,
            std::vector<std::vector<std::complex<double>>> data)
            : m_direction(direction)
            , m_integration_number(integration_number)
            , m_data(data)
        {

        }
    };

    class CalibrationResult
    {
        casacore::MVDirection m_direction;
        std::vector<std::vector<std::complex<double>>> m_data;

    public:
        CalibrationResult(
            casacore::MVDirection direction,
            std::vector<std::vector<std::complex<double>>> data)
            : m_direction(direction)
            , m_data(data)
        {

        }
    };
}