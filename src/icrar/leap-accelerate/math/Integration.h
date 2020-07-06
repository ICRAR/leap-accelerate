
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