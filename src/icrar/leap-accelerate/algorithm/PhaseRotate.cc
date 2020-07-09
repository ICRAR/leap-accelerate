
/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#include "PhaseRotate.h"

#include <icrar/leap-accelerate/math/math.h>
#include <icrar/leap-accelerate/math/casacore_helper.h>

#include <icrar/leap-accelerate/utils.h>
#include <icrar/leap-accelerate/MetaData.h>
#include <icrar/leap-accelerate/math/Integration.h>

//#include <icrar/leap-accelerate/math/matrix.cuh>
#include <icrar/leap-accelerate/math/vector.h>

#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSAntenna.h>

#include <casacore/casa/Quanta/MVDirection.h>
#include <casacore/casa/Quanta/MVuvw.h>

#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>

#include <boost/math/constants/constants.hpp>

#include <istream>
#include <iostream>
#include <iterator>
#include <string>
#include <queue>
#include <optional>
#include <exception>
#include <memory>

using namespace casacore;

using Radians = double;

namespace icrar
{
    // TODO: docs
    // leap_remote_calibration
    // leap_calibrate_from_queue
    std::queue<IntegrationResult> PhaseRotate(MetaData& metadata, const std::vector<casacore::MVDirection>& directions, std::queue<Integration>& input)
    {
        std::queue<IntegrationResult> output = std::queue<IntegrationResult>();
        std::vector<std::vector<std::complex<double>>> cal;

        for(Integration integration = input.front(); !input.empty(); integration = input.front(), input.pop())
        {
            std::function<Radians(std::complex<double>)> getAngle = [](std::complex<double> c) -> Radians
            {
                return std::arg(c);
            };
            casacore::Matrix<Radians> avg_data = MapCollection(metadata.avg_data, getAngle);

            //casacore::Array<double> cal1 = h_multiply(metadata.Ad1, avg_data.column(0));// TODO: (IPosition(0, metadata.I1));
            
            auto a = casacore::Array<double>(IPosition(5));
            auto b = casacore::Array<double>(IPosition(5));
            auto c = casacore::Array<double>(IPosition(5));
            //hello();
            //h_add(a, b, c);

            // auto dInt = casacore::Array<double>(avg_data(IPosition(metadata.I)).shape());
            // for(int n = 0; n < metadata.I; ++n)
            // {
            //     //TODO determine dInt
            //     dInt[n] = avg_data(IPosition(metadata.I)) - metadata.A(IPosition(n)) * cal1;
            // }
            // cal.push_back(icrar::Dot(metadata.Ad, dInt.T[0].T) + cal1);

            // rotateVisibilities(integration, metadata, direction);
            // output.push_back(IntegrationResult(direction, integration.integration_nuumber, std::vector<std::vector<std::complex<double>>>()))
        }

        return output;
    }

    void RotateVisibilities(Integration& integration, MetaData& metadata, const casacore::MVDirection& direction)
    {
        using namespace std::complex_literals;
        auto& data = integration.data;
        auto& uvw = integration.uvw;
        auto parameters = integration.parameters;

        if(metadata.init)
        {
            //metadata['nbaseline']=metadata['stations']*(metadata['stations']-1)/2
            SetDD(metadata, direction);
            SetWv(metadata);
            // Zero a vector for averaging in time and freq
            metadata.avg_data = Matrix<DComplex>(integration.baselines, metadata.num_pols);
            metadata.init = false;
        }
        CalcUVW(uvw, metadata);

        // loop over baselines
        for(int baseline = 0; baseline < integration.baselines; ++baseline)
        {
            // For baseline
            const double pi = boost::math::constants::pi<double>();
            double shiftFactor = -2 * pi * uvw[baseline].get()[2] - metadata.oldUVW[baseline].get()[2]; // check these are correct
            shiftFactor = shiftFactor + 2 * pi * (metadata.phase_centre_ra_rad * metadata.oldUVW[baseline].get()[0]);
            shiftFactor = shiftFactor -2 * pi * (direction.get()[0] * uvw[baseline].get()[0] - direction.get()[1] * uvw[baseline].get()[1]);

            if(baseline % 1000 == 1)
            {
                std::cout << "ShiftFactor for baseline " << baseline << " is " << shiftFactor << std::endl;
            }

            // Loop over channels
            for(int channel = 0; channel < metadata.channels; channel++)
            {
                double shiftRad = shiftFactor / metadata.channel_wavelength[channel];
                double rs = sin(shiftRad);
                double rc = cos(shiftRad);
                std::complex<double> v = data[channel][baseline];

                data[channel][baseline] = v; //TODO * std::exp(1i * shiftRad);
                if(data[channel][baseline].real() == NAN
                || data[channel][baseline].imag() == NAN)
                {
                    metadata.avg_data(IPosition(baseline)) += data[channel][baseline];
                }
            }
        }
    }

    std::pair<Matrix<double>, Array<std::int32_t>> PhaseMatrixFunction(const Array<std::int32_t>& a1, const Array<std::int32_t>& a2, int refAnt, bool map)
    {
        int nAnt = 1 + icrar::Equal(a1,a2) ? 1 : 0;
        if(refAnt >= nAnt - 1)
        {
            throw std::invalid_argument("RefAnt out of bounds");
        }

        Matrix<double> A = Matrix<double>(a1.size() + 1, icrar::ArrayMax(a1));
        for(auto v : A)
        {
            v = 0;
        }

        Matrix<int> I = Matrix<int>(a1.size() + 1, a1.size() + 1);
        for(auto v : I)
        {
            v = 1;
        }

        int k = 0;

        for(int n = 0; n < a1.size(); n++)
        {
            if(a1(IPosition(n)) != a2(IPosition(n)))
            {
                if((refAnt < 0) | ((refAnt >= 0) & ((a1(IPosition(n))==refAnt) | (a2(IPosition(n)) == refAnt))))
                {
                    A(IPosition(k, a1(IPosition(n)))) = 1;
                    A(IPosition(k, a2(IPosition(n)))) = -1;
                    I(IPosition(k)) = n;
                    k++;
                }
            }
        }
        if(refAnt < 0)
        {
            refAnt = 0;
            A(IPosition(k,refAnt)) = 1;
            k++;
            
            throw std::runtime_error("matrix slice needs implementation");
            //A = A[:k];
            //I = I[:k];
        }

        return std::make_pair(A, I);
    }
}
