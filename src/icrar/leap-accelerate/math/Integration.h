
#pragma once

#include <casacore/casa/Quanta/MVuvw.h>

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
}